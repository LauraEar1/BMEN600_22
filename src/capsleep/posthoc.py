from __future__ import annotations

import json
from pathlib import Path
import platform

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from scipy.stats import binomtest, chi2_contingency
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder

from .analysis_cohort import _read_json
from .dataset import dataset_digest


MODEL_NAMES = ("eeg", "eeg_eog_emg", "eeg_eog_emg_ecg")
MODEL_PAIRS = (
    ("eeg", "eeg_eog_emg"),
    ("eeg_eog_emg", "eeg_eog_emg_ecg"),
    ("eeg", "eeg_eog_emg_ecg"),
)
MODALITIES = ("eeg", "emg", "eog", "ecg")
IDENTIFIERS = ("record_id", "subject_id", "label", "fold")


def align_baseline_predictions(
    assignments: pd.DataFrame,
    predictions: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    required = set(IDENTIFIERS)
    if not required.issubset(assignments.columns):
        raise ValueError(f"Fold assignments missing columns: {sorted(required - set(assignments.columns))}")
    if assignments["record_id"].duplicated().any() or assignments["subject_id"].duplicated().any():
        raise ValueError("Fold assignments must contain unique recording and subject IDs")
    aligned = assignments.loc[:, IDENTIFIERS].copy().reset_index(drop=True)
    for model_name in MODEL_NAMES:
        if model_name not in predictions:
            raise ValueError(f"Missing out-of-fold predictions for {model_name}")
        frame = predictions[model_name]
        required_prediction_columns = required | {"prediction"}
        if not required_prediction_columns.issubset(frame.columns):
            raise ValueError(
                f"{model_name} predictions missing columns: "
                f"{sorted(required_prediction_columns - set(frame.columns))}"
            )
        if frame["record_id"].duplicated().any() or frame["subject_id"].duplicated().any():
            raise ValueError(f"{model_name} predictions contain duplicate identifiers")
        if set(frame["record_id"]) != set(aligned["record_id"]):
            raise ValueError(f"{model_name} prediction recording IDs do not match assignments")
        comparison = aligned.merge(
            frame.loc[:, [*IDENTIFIERS, "prediction"]],
            on="record_id",
            how="left",
            suffixes=("_assignment", "_prediction"),
            validate="one_to_one",
            sort=False,
        )
        for column in ("subject_id", "label", "fold"):
            if not comparison[f"{column}_assignment"].equals(
                comparison[f"{column}_prediction"]
            ):
                raise ValueError(
                    f"{model_name} prediction {column} differs from fold assignments"
                )
        aligned[f"{model_name}_prediction"] = comparison["prediction"].to_numpy()
        aligned[f"{model_name}_correct"] = (
            comparison["prediction"].to_numpy() == aligned["label"].to_numpy()
        )
    return aligned


def paired_correctness_changes(aligned: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for first, second in MODEL_PAIRS:
        first_correct = aligned[f"{first}_correct"].astype(bool)
        second_correct = aligned[f"{second}_correct"].astype(bool)
        for index, row in aligned.iterrows():
            a = bool(first_correct.iloc[index])
            b = bool(second_correct.iloc[index])
            rows.append({
                "comparison": f"{first}_vs_{second}",
                "record_id": row["record_id"],
                "subject_id": row["subject_id"],
                "label": row["label"],
                "fold": int(row["fold"]),
                "first_model": first,
                "second_model": second,
                "first_correct": a,
                "second_correct": b,
                "change": (
                    "both_correct" if a and b else
                    "both_incorrect" if not a and not b else
                    f"{second}_improved" if b else f"{second}_regressed"
                ),
            })
    return pd.DataFrame(rows)


def exact_mcnemar(aligned: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for first, second in MODEL_PAIRS:
        a = aligned[f"{first}_correct"].astype(bool).to_numpy()
        b = aligned[f"{second}_correct"].astype(bool).to_numpy()
        first_only = int(np.count_nonzero(a & ~b))
        second_only = int(np.count_nonzero(~a & b))
        discordant = first_only + second_only
        p_value = (
            float(binomtest(min(first_only, second_only), discordant, 0.5).pvalue)
            if discordant else 1.0
        )
        rows.append({
            "comparison": f"{first}_vs_{second}",
            "first_model": first,
            "second_model": second,
            "n_subjects": len(aligned),
            "both_correct": int(np.count_nonzero(a & b)),
            "both_incorrect": int(np.count_nonzero(~a & ~b)),
            "first_correct_second_incorrect": first_only,
            "first_incorrect_second_correct": second_only,
            "first_accuracy": float(a.mean()),
            "second_accuracy": float(b.mean()),
            "accuracy_difference_second_minus_first": float(b.mean() - a.mean()),
            "mcnemar_exact_two_sided_p": p_value,
            "interpretation": "exploratory paired test; not adjusted for multiple comparisons",
        })
    return pd.DataFrame(rows)


def _metric_arrays(truth: np.ndarray, prediction: np.ndarray, labels: list[str]):
    return (
        balanced_accuracy_score(truth, prediction),
        f1_score(truth, prediction, labels=labels, average="macro", zero_division=0),
    )


def paired_stratified_bootstrap(
    aligned: pd.DataFrame,
    first: str,
    second: str,
    labels: list[str],
    repetitions: int = 10000,
    random_seed: int = 42,
) -> dict:
    if repetitions < 100:
        raise ValueError("At least 100 bootstrap repetitions are required")
    truth = aligned["label"].to_numpy()
    first_prediction = aligned[f"{first}_prediction"].to_numpy()
    second_prediction = aligned[f"{second}_prediction"].to_numpy()
    groups = [
        np.flatnonzero(truth == label)
        for label in labels
    ]
    if any(len(group) == 0 for group in groups) or set(truth) != set(labels):
        raise ValueError(
            "Paired stratified bootstrap requires exactly the configured classes"
        )
    rng = np.random.default_rng(random_seed)
    weights = np.zeros((repetitions, len(truth)), dtype=np.int16)
    for group in groups:
        weights[:, group] = rng.multinomial(
            len(group),
            np.full(len(group), 1 / len(group)),
            size=repetitions,
        )

    def bootstrap_scores(prediction: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        confusion = np.zeros(
            (repetitions, len(labels), len(labels)), dtype=np.int32
        )
        for true_index, label in enumerate(labels):
            mask = truth == label
            group_weights = weights[:, mask]
            group_predictions = prediction[mask]
            for predicted_index, predicted_label in enumerate(labels):
                confusion[:, true_index, predicted_index] = group_weights[
                    :, group_predictions == predicted_label
                ].sum(axis=1)
        true_support = confusion.sum(axis=2)
        predicted_support = confusion.sum(axis=1)
        diagonal = np.diagonal(confusion, axis1=1, axis2=2)
        balanced_accuracy = np.mean(diagonal / true_support, axis=1)
        f1_denominator = true_support + predicted_support
        class_f1 = np.divide(
            2 * diagonal,
            f1_denominator,
            out=np.zeros_like(diagonal, dtype=float),
            where=f1_denominator != 0,
        )
        return balanced_accuracy, np.mean(class_f1, axis=1)

    first_distributions = bootstrap_scores(first_prediction)
    second_distributions = bootstrap_scores(second_prediction)
    point_first = _metric_arrays(truth, first_prediction, labels)
    point_second = _metric_arrays(truth, second_prediction, labels)
    distributions = {
        "balanced_accuracy": (
            second_distributions[0] - first_distributions[0],
            point_second[0] - point_first[0],
        ),
        "macro_f1": (
            second_distributions[1] - first_distributions[1],
            point_second[1] - point_first[1],
        ),
    }
    output = {}
    for metric, (array, point_difference) in distributions.items():
        output[metric] = {
            "difference_second_minus_first": float(point_difference),
            "ci_95_percentile": [
                float(np.percentile(array, 2.5)),
                float(np.percentile(array, 97.5)),
            ],
        }
    return {
        "comparison": f"{first}_vs_{second}",
        "method": "paired subject bootstrap stratified within diagnostic class",
        "repetitions": repetitions,
        "seed": random_seed,
        "metrics": output,
        "interpretation": "exploratory intervals; do not imply independent windows or confirmatory inference",
    }


def calculate_missingness(
    features: pd.DataFrame,
    modality_columns: dict[str, list[str]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    per_record = []
    for record in features.loc[:, ["record_id", "subject_id", "label"]].to_dict("records"):
        values_by_modality = features.loc[
            features["record_id"].eq(record["record_id"])
        ]
        for modality, columns in modality_columns.items():
            values = values_by_modality.loc[:, columns]
            cells = int(values.size)
            missing = int(values.isna().sum().sum())
            per_record.append({
                "record_id": record["record_id"],
                "subject_id": record["subject_id"],
                "label": record["label"],
                "modality": modality,
                "feature_count": len(columns),
                "missing_cells": missing,
                "total_cells": cells,
                "missing_fraction": missing / cells if cells else float("nan"),
                "all_missing_feature_count": int(values.isna().all(axis=0).sum()),
            })
    record_table = pd.DataFrame(per_record)
    class_table = (
        record_table.groupby(["label", "modality"], observed=True)
        .agg(
            subject_count=("record_id", "nunique"),
            feature_count=("feature_count", "first"),
            missing_cells=("missing_cells", "sum"),
            total_cells=("total_cells", "sum"),
            all_missing_record_modality_pairs=("all_missing_feature_count", "sum"),
        )
        .reset_index()
    )
    class_table["missing_fraction"] = (
        class_table["missing_cells"] / class_table["total_cells"]
    )
    return record_table, class_table


def _make_model(columns: list[str], trees: int, seed: int):
    return make_pipeline(
        SimpleImputer(strategy="median", keep_empty_features=True),
        RandomForestClassifier(
            n_estimators=trees,
            class_weight="balanced_subsample",
            random_state=seed,
            n_jobs=-1,
        ),
    )


def _empty_feature_counts(
    train: pd.DataFrame, columns: list[str]
) -> dict[str, int]:
    counts = {}
    for modality in MODALITIES:
        modality_columns = [
            column for column in columns if column.startswith(f"{modality}__")
        ]
        counts[modality] = int(
            train[modality_columns].isna().all(axis=0).sum()
        ) if modality_columns else 0
    counts["all_modalities"] = sum(counts.values())
    return counts


def _harmonized_columns(
    train: pd.DataFrame,
    columns: list[str],
) -> list[str]:
    return [column for column in columns if not train[column].isna().all()]


def _confusion_metrics(
    truth: np.ndarray, prediction: np.ndarray, labels: list[str]
) -> tuple[float, float]:
    return (
        float(balanced_accuracy_score(truth, prediction)),
        float(f1_score(truth, prediction, labels=labels, average="macro", zero_division=0)),
    )


def _acquisition_features(settings, record_ids: list[str]) -> pd.DataFrame:
    rows = []
    record_dir = settings.root / "results" / "checkpoints" / "records"
    for record_id in record_ids:
        payload = _read_json(record_dir / f"{record_id}.json")
        channels = payload["channels"]
        row = {"record_id": record_id}
        for modality in MODALITIES:
            selected = [
                channel for channel in channels
                if channel.get("modality") == modality
            ]
            row[f"{modality}_channel_count"] = len(selected)
            row[f"{modality}_native_rates_hz"] = ";".join(
                str(value) for value in sorted({
                    float(channel["native_sfreq_hz"]) for channel in selected
                })
            )
            row[f"{modality}_channel_labels"] = ";".join(sorted(
                channel.get("feature_channel", channel["channel"])
                for channel in selected
            ))
        rows.append(row)
    return pd.DataFrame(rows)


def _cramers_v_table(
    metadata: pd.DataFrame,
    labels: pd.Series,
    columns: list[str],
) -> pd.DataFrame:
    rows = []
    for column in columns:
        table = pd.crosstab(metadata[column].fillna("<missing>"), labels)
        if table.shape[0] < 2 or table.shape[1] < 2:
            rows.append({
                "acquisition_variable": column,
                "levels": table.shape[0],
                "cramers_v": 0.0,
                "chi_square_p_unadjusted": 1.0,
            })
            continue
        chi_square, p_value, _, _ = chi2_contingency(table, correction=False)
        n = int(table.to_numpy().sum())
        denominator = n * min(table.shape[0] - 1, table.shape[1] - 1)
        rows.append({
            "acquisition_variable": column,
            "levels": table.shape[0],
            "cramers_v": float(np.sqrt(chi_square / denominator)) if denominator else 0.0,
            "chi_square_p_unadjusted": float(p_value),
        })
    return pd.DataFrame(rows)


def _make_figures(
    output_dir: Path,
    baseline_summary: pd.DataFrame,
    importance_summary: pd.DataFrame,
    class_missingness: pd.DataFrame,
) -> None:
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    fig, axis = plt.subplots(figsize=(8, 4.5))
    models = list(baseline_summary["model"])
    positions = np.arange(len(models))
    width = 0.35
    axis.bar(
        positions - width / 2,
        baseline_summary["balanced_accuracy"],
        width,
        label="Baseline OOF",
    )
    axis.bar(
        positions + width / 2,
        baseline_summary["harmonized_balanced_accuracy"],
        width,
        label="Train-supported sensitivity",
    )
    axis.set_xticks(positions, models, rotation=15, ha="right")
    axis.set_ylabel("Balanced accuracy")
    axis.set_ylim(0, 1)
    axis.legend()
    fig.tight_layout()
    fig.savefig(figure_dir / "baseline_vs_harmonized_balanced_accuracy.png", dpi=160)
    plt.close(fig)

    heatmap = class_missingness.pivot(
        index="label", columns="modality", values="missing_fraction"
    ).reindex(columns=list(MODALITIES))
    fig, axis = plt.subplots(figsize=(7, 4))
    image = axis.imshow(heatmap.to_numpy(), vmin=0, vmax=1, cmap="magma", aspect="auto")
    axis.set_xticks(np.arange(len(heatmap.columns)), heatmap.columns)
    axis.set_yticks(np.arange(len(heatmap.index)), heatmap.index)
    axis.set_title("Subject-level feature missingness by class and modality")
    fig.colorbar(image, ax=axis, label="Fraction missing")
    fig.tight_layout()
    fig.savefig(figure_dir / "missingness_by_class_modality.png", dpi=160)
    plt.close(fig)

    if not importance_summary.empty:
        modalities = list(MODALITIES)
        models = list(MODEL_NAMES)
        width = 0.8 / len(models)
        fig, axis = plt.subplots(figsize=(9, 4.5))
        for model_index, model in enumerate(models):
            summary = (
                importance_summary.loc[importance_summary["model"].eq(model)]
                .set_index("modality")
                .reindex(modalities)
            )
            axis.bar(
                np.arange(len(modalities)) + (model_index - (len(models) - 1) / 2) * width,
                summary["balanced_accuracy_drop_mean"],
                width,
                yerr=summary["balanced_accuracy_drop_sd"].fillna(0),
                capsize=3,
                label=model,
            )
        axis.set_xticks(np.arange(len(modalities)), modalities)
        axis.axhline(0, color="black", linewidth=0.8)
        axis.set_ylabel("Held-out balanced-accuracy drop")
        axis.set_title("Exploratory grouped modality permutation importance")
        axis.legend()
        fig.tight_layout()
        fig.savefig(figure_dir / "grouped_modality_importance.png", dpi=160)
        plt.close(fig)


def run_posthoc_primary(
    settings,
    repetitions: int = 10000,
    permutation_repetitions: int = 10,
) -> dict:
    cohort_config_path = (
        settings.root / "config" / "analysis_cohorts"
        / "primary_physiology_84_v2.json"
    )
    cohort_config = _read_json(cohort_config_path)
    analysis_id = cohort_config["analysis_id"]
    baseline_dir = settings.root / "results" / "models" / analysis_id
    baseline_metrics_path = baseline_dir / "metrics.json"
    assignment_path = baseline_dir / "fold_assignments.csv"
    feature_path = (
        settings.root / "results" / "features"
        / "physiology_primary_84_v2_candidate.csv.gz"
    )
    if not all(path.is_file() for path in (
        baseline_metrics_path, assignment_path, feature_path
    )):
        raise FileNotFoundError("Required v2 baseline metrics, folds, or features are missing")
    metrics = _read_json(baseline_metrics_path)
    if metrics.get("analysis_id") != analysis_id:
        raise ValueError("Baseline metrics belong to a different analysis cohort")
    if metrics.get("reproducibility", {}).get(
        "subject_feature_dataset_sha256"
    ) != dataset_digest(feature_path):
        raise ValueError("Subject feature dataset differs from the saved baseline evaluation")
    expected_config_hash = metrics.get("reproducibility", {}).get(
        "analysis_cohort_config_sha256"
    )
    if expected_config_hash != dataset_digest(cohort_config_path):
        raise ValueError("Analysis cohort config differs from the saved baseline evaluation")
    assignments = pd.read_csv(assignment_path)
    predictions = {
        model_name: pd.read_csv(
            baseline_dir / f"{model_name}_oof_predictions.csv"
        )
        for model_name in MODEL_NAMES
    }
    aligned = align_baseline_predictions(assignments, predictions)
    features = pd.read_csv(feature_path)
    if features["record_id"].tolist() != assignments["record_id"].tolist():
        raise ValueError("Subject feature order differs from the saved fold assignments")
    if features["subject_id"].tolist() != assignments["subject_id"].tolist():
        raise ValueError("Subject IDs in features differ from saved folds")
    if features["label"].tolist() != assignments["label"].tolist():
        raise ValueError("True labels in features differ from saved folds")
    labels = sorted(cohort_config["expected_class_counts"])
    for model_name in MODEL_NAMES:
        saved_columns = metrics["models"][model_name]["feature_columns"]
        if not set(saved_columns).issubset(features.columns):
            raise ValueError(f"Baseline feature list is missing from subject table: {model_name}")
    n = len(features)
    if n != len(cohort_config["included_record_ids"]):
        raise ValueError("Post-hoc participant count differs from the approved cohort")

    output_dir = settings.root / "results" / "analysis" / "posthoc_primary_84_v2"
    output_dir.mkdir(parents=True, exist_ok=True)
    # Save the immutable baseline alignment as a traceable copy in the new analysis directory.
    aligned.to_csv(output_dir / "aligned_baseline_oof_predictions.csv", index=False)
    correctness = paired_correctness_changes(aligned)
    correctness.to_csv(output_dir / "participant_correctness_changes.csv", index=False)
    mcnemar = exact_mcnemar(aligned)
    mcnemar.to_csv(output_dir / "mcnemar_exact_tests.csv", index=False)

    bootstrap_rows = []
    for pair_index, (first, second) in enumerate(MODEL_PAIRS):
        result = paired_stratified_bootstrap(
            aligned,
            first,
            second,
            labels,
            repetitions=repetitions,
            random_seed=int(cohort_config["primary_model"]["random_seed"]) + pair_index,
        )
        for metric, values in result["metrics"].items():
            bootstrap_rows.append({
                "comparison": result["comparison"],
                "metric": metric,
                "difference_second_minus_first": values["difference_second_minus_first"],
                "ci_95_low": values["ci_95_percentile"][0],
                "ci_95_high": values["ci_95_percentile"][1],
                "bootstrap_repetitions": repetitions,
                "bootstrap_seed": result["seed"],
                "method": result["method"],
                "interpretation": result["interpretation"],
            })
    bootstrap_table = pd.DataFrame(bootstrap_rows)
    bootstrap_table.to_csv(
        output_dir / "paired_subject_bootstrap_intervals.csv", index=False
    )

    modality_columns = {
        modality: [
            column for column in features.columns
            if column.startswith(f"{modality}__")
        ]
        for modality in MODALITIES
    }
    record_missingness, class_missingness = calculate_missingness(
        features, modality_columns
    )
    record_missingness.to_csv(output_dir / "missingness_by_record_modality.csv", index=False)
    class_missingness.to_csv(output_dir / "missingness_by_class_modality.csv", index=False)

    split_by_fold = {
        fold: (
            np.flatnonzero(assignments["fold"].to_numpy() != fold),
            np.flatnonzero(assignments["fold"].to_numpy() == fold),
        )
        for fold in sorted(assignments["fold"].unique())
    }
    imputation_rows = []
    imputation_detail_rows = []
    baseline_replication_rows = []
    importance_rows = []
    harmonized_predictions = {}
    harmonized_feature_counts = {}
    tree_count = int(metrics["n_estimators"])
    seed = int(metrics["random_seed"])

    for model_name in MODEL_NAMES:
        columns = metrics["models"][model_name]["feature_columns"]
        prediction = np.empty(n, dtype=object)
        fold_selected_counts = []
        for fold, (train_indices, test_indices) in split_by_fold.items():
            train = features.iloc[train_indices]
            test = features.iloc[test_indices]
            model = _make_model(columns, tree_count, seed)
            model.fit(train[columns], train["label"])
            baseline_pred = model.predict(test[columns])
            saved = aligned.set_index("record_id").loc[
                test["record_id"], f"{model_name}_prediction"
            ].to_numpy()
            if not np.array_equal(baseline_pred, saved):
                raise ValueError(
                    f"Recreated {model_name} predictions differ from saved baseline in fold {fold}"
                )
            prediction[test_indices] = baseline_pred
            baseline_replication_rows.append({
                "model": model_name,
                "fold": int(fold),
                "prediction_match": True,
                "test_subject_count": len(test_indices),
            })

            empty_counts = _empty_feature_counts(train, columns)
            imputer = model.named_steps["simpleimputer"]
            empty_mask = train[columns].isna().all(axis=0)
            empty_feature_names = empty_mask.index[empty_mask].tolist()
            empty_indices = empty_mask.to_numpy()
            empty_statistics = imputer.statistics_[empty_indices]
            test_empty_values = test[columns].to_numpy()[:, empty_indices]
            test_empty_transformed = imputer.transform(
                test[columns]
            )[:, empty_indices]
            test_missing_values = np.isnan(test_empty_values)
            zero_imputation_verified = bool(
                np.allclose(empty_statistics, 0.0)
                and np.allclose(
                    imputer.transform(train[columns])[:, empty_indices], 0.0
                )
                and np.allclose(
                    test_empty_transformed[test_missing_values], 0.0
                )
                and np.allclose(
                    test_empty_transformed[~test_missing_values],
                    test_empty_values[~test_missing_values],
                )
            )
            if empty_counts["all_modalities"] and not zero_imputation_verified:
                raise ValueError(
                    "The saved baseline imputer treatment of training-empty columns "
                    f"could not be verified for {model_name}, fold {fold}"
                )
            imputation_rows.append({
                "model": model_name,
                "fold": int(fold),
                **{
                    f"training_all_missing_{modality}_features": empty_counts[modality]
                    for modality in MODALITIES
                },
                "training_all_missing_total_features": empty_counts["all_modalities"],
                "imputer_strategy": imputer.strategy,
                "keep_empty_features": imputer.keep_empty_features,
                "empty_features_imputed_to_zero_verified": zero_imputation_verified,
                "empty_features_retained_in_random_forest": empty_counts["all_modalities"],
            })
            imputation_detail_rows.extend({
                "model": model_name,
                "fold": int(fold),
                "feature": column,
                "imputer_strategy": imputer.strategy,
                "imputed_value": float(imputer.statistics_[columns.index(column)]),
                "retained_in_random_forest": True,
            } for column in empty_feature_names)

            truth = test["label"].to_numpy()
            base_balanced, base_f1 = _confusion_metrics(truth, baseline_pred, labels)
            for modality in MODALITIES:
                group_columns = [
                    column for column in columns
                    if column.startswith(f"{modality}__")
                ]
                if not group_columns:
                    continue
                source = test[group_columns].to_numpy(copy=True)
                for repeat in range(permutation_repetitions):
                    rng = np.random.default_rng(
                        seed + int(fold) * 1009 + repeat * 9176
                        + MODALITIES.index(modality) * 131
                        + MODEL_NAMES.index(model_name) * 65537
                    )
                    order = rng.permutation(len(test_indices))
                    permuted = test[columns].copy()
                    permuted.loc[:, group_columns] = source[order, :]
                    permuted_prediction = model.predict(permuted)
                    perm_balanced, perm_f1 = _confusion_metrics(
                        truth, permuted_prediction, labels
                    )
                    importance_rows.append({
                        "model": model_name,
                        "fold": int(fold),
                        "modality": modality,
                        "repeat": repeat,
                        "feature_count_permuted_together": len(group_columns),
                        "baseline_balanced_accuracy": base_balanced,
                        "permuted_balanced_accuracy": perm_balanced,
                        "balanced_accuracy_drop": base_balanced - perm_balanced,
                        "baseline_macro_f1": base_f1,
                        "permuted_macro_f1": perm_f1,
                        "macro_f1_drop": base_f1 - perm_f1,
                    })

            harmonized = _harmonized_columns(train, columns)
            if not harmonized:
                raise ValueError(
                    f"No training-supported features for {model_name}, fold {fold}"
                )
            fold_selected_counts.append(len(harmonized))
            harmonized_model = _make_model(harmonized, tree_count, seed)
            harmonized_model.fit(train[harmonized], train["label"])
            if model_name not in harmonized_predictions:
                harmonized_predictions[model_name] = np.empty(n, dtype=object)
            harmonized_predictions[model_name][test_indices] = (
                harmonized_model.predict(test[harmonized])
            )
        harmonized_feature_counts[model_name] = fold_selected_counts
    pd.DataFrame(baseline_replication_rows).to_csv(
        output_dir / "baseline_prediction_reproduction.csv", index=False
    )
    pd.DataFrame(imputation_rows).to_csv(
        output_dir / "training_fold_empty_feature_imputation.csv", index=False
    )
    pd.DataFrame(imputation_detail_rows, columns=[
        "model", "fold", "feature", "imputer_strategy", "imputed_value",
        "retained_in_random_forest",
    ]).to_csv(
        output_dir / "training_fold_empty_feature_details.csv", index=False
    )
    importance = pd.DataFrame(importance_rows)
    importance.to_csv(output_dir / "grouped_modality_permutation_by_fold.csv", index=False)
    if not importance.empty:
        importance_summary = (
            importance.groupby(["model", "modality"], observed=True)
            .agg(
                feature_count_permuted_together=("feature_count_permuted_together", "first"),
                balanced_accuracy_drop_mean=("balanced_accuracy_drop", "mean"),
                balanced_accuracy_drop_sd=("balanced_accuracy_drop", "std"),
                macro_f1_drop_mean=("macro_f1_drop", "mean"),
                macro_f1_drop_sd=("macro_f1_drop", "std"),
                fold_count=("fold", "nunique"),
            )
            .reset_index()
        )
    else:
        importance_summary = pd.DataFrame(columns=[
            "model", "modality", "feature_count_permuted_together",
            "balanced_accuracy_drop_mean", "balanced_accuracy_drop_sd",
            "macro_f1_drop_mean", "macro_f1_drop_sd", "fold_count",
        ])
    importance_summary.to_csv(
        output_dir / "grouped_modality_permutation_summary.csv", index=False
    )

    baseline_summary = []
    harmonized_rows = []
    for model_name in MODEL_NAMES:
        original = aligned[f"{model_name}_prediction"].to_numpy()
        sensitivity = harmonized_predictions[model_name]
        original_scores = _confusion_metrics(
            aligned["label"].to_numpy(), original, labels
        )
        sensitivity_scores = _confusion_metrics(
            aligned["label"].to_numpy(), sensitivity, labels
        )
        baseline_summary.append({
            "model": model_name,
            "balanced_accuracy": original_scores[0],
            "macro_f1": original_scores[1],
            "harmonized_balanced_accuracy": sensitivity_scores[0],
            "harmonized_macro_f1": sensitivity_scores[1],
        })
        for index, row in assignments.iterrows():
            harmonized_rows.append({
                "model": model_name,
                "record_id": row["record_id"],
                "subject_id": row["subject_id"],
                "label": row["label"],
                "fold": int(row["fold"]),
                "prediction": sensitivity[index],
                "correct": bool(sensitivity[index] == row["label"]),
            })
    baseline_summary_frame = pd.DataFrame(baseline_summary)
    baseline_summary_frame.to_csv(
        output_dir / "harmonized_sensitivity_summary.csv", index=False
    )
    pd.DataFrame(harmonized_rows).to_csv(
        output_dir / "harmonized_sensitivity_oof_predictions.csv", index=False
    )
    pd.DataFrame([
        {
            "model": model_name,
            "fold": int(fold),
            "training_supported_feature_count": count,
        }
        for model_name, counts in harmonized_feature_counts.items()
        for fold, count in zip(
            split_by_fold.keys(), counts, strict=True
        )
    ]).to_csv(output_dir / "harmonized_training_feature_counts.csv", index=False)

    acquisition = _acquisition_features(
        settings, assignments["record_id"].tolist()
    )
    acquisition = assignments[["record_id", "subject_id", "label", "fold"]].merge(
        acquisition, on="record_id", validate="one_to_one"
    )
    acquisition.to_csv(output_dir / "acquisition_metadata_by_record.csv", index=False)
    categorical_acquisition_columns = [
        column for column in acquisition.columns
        if column.endswith("_native_rates_hz") or column.endswith("_channel_labels")
    ]
    acquisition_association = _cramers_v_table(
        acquisition, acquisition["label"], categorical_acquisition_columns
    )
    acquisition_association.to_csv(
        output_dir / "acquisition_class_association.csv", index=False
    )
    acquisition_only_rows = []
    numeric_columns = [
        column for column in acquisition.columns
        if column.endswith("_channel_count")
    ]
    preprocessing = ColumnTransformer(
        [
            ("numeric", SimpleImputer(strategy="median"), numeric_columns),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_acquisition_columns,
            ),
        ]
    )
    for fold, (train_indices, test_indices) in split_by_fold.items():
        diagnostic = make_pipeline(
            preprocessing,
            RandomForestClassifier(
                n_estimators=300,
                class_weight="balanced_subsample",
                random_state=seed,
                n_jobs=-1,
            ),
        )
        diagnostic.fit(
            acquisition.iloc[train_indices][numeric_columns + categorical_acquisition_columns],
            acquisition.iloc[train_indices]["label"],
        )
        prediction = diagnostic.predict(
            acquisition.iloc[test_indices][numeric_columns + categorical_acquisition_columns]
        )
        truth = acquisition.iloc[test_indices]["label"].to_numpy()
        balanced, macro = _confusion_metrics(truth, prediction, labels)
        acquisition_only_rows.append({
            "fold": int(fold),
            "test_subject_count": len(test_indices),
            "balanced_accuracy": balanced,
            "macro_f1": macro,
            "test_class_counts": json.dumps(
                acquisition.iloc[test_indices]["label"].value_counts().to_dict(),
                sort_keys=True,
            ),
        })
    acquisition_only = pd.DataFrame(acquisition_only_rows)
    acquisition_only.to_csv(
        output_dir / "acquisition_only_cv_by_fold.csv", index=False
    )
    acquisition_only_summary = {
        metric: {
            "mean_across_folds": float(acquisition_only[metric].mean()),
            "standard_deviation_across_folds": float(
                acquisition_only[metric].std(ddof=1)
            ),
        }
        for metric in ("balanced_accuracy", "macro_f1")
    }

    _make_figures(
        output_dir, baseline_summary_frame, importance_summary, class_missingness
    )
    report = {
        "analysis": "posthoc_primary_84_v2",
        "status": "complete",
        "exploratory": True,
        "baseline_outputs_modified": False,
        "cap_annotations_used": False,
        "participant_count": n,
        "class_counts": features["label"].value_counts().to_dict(),
        "subject_feature_count": len(set().union(*(
            metrics["models"][model_name]["feature_columns"]
            for model_name in MODEL_NAMES
        ))),
        "feature_counts_by_model": {
            model_name: len(metrics["models"][model_name]["feature_columns"])
            for model_name in MODEL_NAMES
        },
        "feature_counts_by_modality": {
            modality: len(columns)
            for modality, columns in modality_columns.items()
        },
        "baseline_predictions_reproduced_exactly": True,
        "baseline_metrics_sha256": dataset_digest(baseline_metrics_path),
        "baseline_fold_assignments_sha256": dataset_digest(assignment_path),
        "baseline_feature_dataset_sha256": dataset_digest(feature_path),
        "paired_tests": "exact two-sided McNemar test on paired subject correctness; unadjusted exploratory p-values",
        "bootstrap": {
            "method": "paired subject bootstrap, stratified within true diagnostic class",
            "repetitions": repetitions,
            "seeds": [
                int(cohort_config["primary_model"]["random_seed"]) + index
                for index in range(len(MODEL_PAIRS))
            ],
            "confidence_interval": "percentile 95%",
        },
        "harmonized_sensitivity": {
            "definition": (
                "Within each fold and model, retain features that are not entirely "
                "missing among training subjects; the test fold never determines "
                "the selected feature list. Refit the same Random Forest settings "
                "and training-fold median imputation."
            ),
            "selected_feature_counts_by_model_fold": harmonized_feature_counts,
        },
        "grouped_modality_importance": {
            "method": (
                "Within each held-out fold, jointly permute all test rows for one "
                "modality's feature block using a shared row permutation; report "
                "paired drop from unpermuted held-out metrics."
            ),
            "permutation_repetitions_per_fold_modality": permutation_repetitions,
            "interpretation": "exploratory and potentially noisy with small test folds and correlated features",
        },
        "acquisition_confounding": {
            "metadata_only_model": "one-hot acquisition channel labels and native-rate sets plus per-modality channel counts, evaluated on saved folds",
            "metadata_only_cv_summary": acquisition_only_summary,
            "interpretation": "exploratory diagnostic of acquisition-label association, not a clinical classifier",
        },
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
        },
    }
    (output_dir / "posthoc_report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return report
