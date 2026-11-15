from __future__ import annotations

import json
import platform
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import binomtest

from .dataset import dataset_digest


MODEL_ORDER = ("eeg", "eeg_eog_emg", "eeg_eog_emg_ecg")
MODEL_LABELS = {
    "eeg": "EEG",
    "eeg_eog_emg": "EEG + EMG + EOG",
    "eeg_eog_emg_ecg": "EEG + EMG + EOG + ECG",
}
MODEL_COLORS = {
    "eeg": "#0072B2",
    "eeg_eog_emg": "#009E73",
    "eeg_eog_emg_ecg": "#D55E00",
}
CLASS_ORDER = ("control", "nfle", "rbd", "plm")
CLASS_LABELS = {
    "control": "Healthy controls",
    "nfle": "NFLE",
    "rbd": "RBD",
    "plm": "PLM",
}
MODALITY_ORDER = ("eeg", "emg", "eog", "ecg")
MODALITY_LABELS = {
    "eeg": "EEG",
    "emg": "EMG",
    "eog": "EOG",
    "ecg": "ECG",
}


@dataclass(frozen=True)
class ReportInputs:
    root: Path
    assignments: pd.DataFrame
    predictions: dict[str, pd.DataFrame]
    metrics: dict
    confusion_matrices: dict[str, np.ndarray]
    harmonized_predictions: pd.DataFrame
    harmonized_summary: pd.DataFrame
    bootstrap: pd.DataFrame
    mcnemar: pd.DataFrame
    correctness_changes: pd.DataFrame
    missingness: pd.DataFrame
    acquisition_folds: pd.DataFrame
    acquisition_associations: pd.DataFrame
    acquisition_metadata: pd.DataFrame
    cohort_config: dict
    source_manifest: dict
    posthoc_report: dict
    sources: dict[str, Path]


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _require_columns(frame: pd.DataFrame, columns: set[str], context: str) -> None:
    missing = columns - set(frame.columns)
    if missing:
        raise ValueError(f"{context} is missing required columns: {sorted(missing)}")


def validate_prediction_alignment(
    assignments: pd.DataFrame,
    predictions: dict[str, pd.DataFrame],
    expected_record_ids: set[str] | None = None,
) -> dict[str, pd.DataFrame]:
    identifiers = ["record_id", "subject_id", "label", "fold"]
    _require_columns(assignments, set(identifiers), "Fold assignments")
    if assignments["record_id"].duplicated().any():
        raise ValueError("Fold assignments contain duplicate recording IDs")
    if assignments["subject_id"].duplicated().any():
        raise ValueError("Fold assignments contain duplicate subject IDs")
    assigned_ids = set(assignments["record_id"].astype(str))
    if expected_record_ids is not None and assigned_ids != expected_record_ids:
        raise ValueError("Fold assignments do not match the approved cohort IDs")

    aligned: dict[str, pd.DataFrame] = {}
    reference = assignments.set_index("record_id")[["subject_id", "label", "fold"]]
    for model in MODEL_ORDER:
        if model not in predictions:
            raise ValueError(f"Missing out-of-fold predictions for model {model}")
        frame = predictions[model].copy()
        _require_columns(frame, {*identifiers, "prediction"}, f"{model} predictions")
        if frame["record_id"].duplicated().any() or frame["subject_id"].duplicated().any():
            raise ValueError(f"{model} predictions contain duplicate identifiers")
        if set(frame["record_id"].astype(str)) != assigned_ids:
            raise ValueError(f"{model} prediction recording IDs differ from assignments")
        indexed = frame.set_index("record_id").loc[assignments["record_id"]]
        for column in ("subject_id", "label", "fold"):
            expected = reference[column].to_numpy()
            observed = indexed[column].to_numpy()
            if not np.array_equal(expected, observed):
                raise ValueError(
                    f"{model} prediction {column} differs from fold assignments"
                )
        aligned[model] = indexed.reset_index()
    return aligned


def _metrics_from_predictions(
    truth: np.ndarray,
    predicted: np.ndarray,
    labels: tuple[str, ...] = CLASS_ORDER,
) -> tuple[float, float, np.ndarray, dict[str, dict[str, float]]]:
    matrix = np.zeros((len(labels), len(labels)), dtype=int)
    label_index = {label: index for index, label in enumerate(labels)}
    for actual, guess in zip(truth, predicted, strict=True):
        if actual not in label_index or guess not in label_index:
            raise ValueError("Prediction contains a label outside the approved classes")
        matrix[label_index[actual], label_index[guess]] += 1
    support = matrix.sum(axis=1)
    if np.any(support == 0):
        raise ValueError("At least one diagnostic class is absent from predictions")
    recall = np.diag(matrix) / support
    predicted_support = matrix.sum(axis=0)
    precision = np.divide(
        np.diag(matrix),
        predicted_support,
        out=np.zeros(len(labels), dtype=float),
        where=predicted_support > 0,
    )
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros(len(labels), dtype=float),
        where=(precision + recall) > 0,
    )
    per_class = {
        label: {
            "recall": float(recall[index]),
            "f1": float(f1[index]),
            "support": float(support[index]),
        }
        for index, label in enumerate(labels)
    }
    return float(recall.mean()), float(f1.mean()), matrix, per_class


def _validate_baseline_metrics(
    metrics: dict,
    assignments: pd.DataFrame,
    predictions: dict[str, pd.DataFrame],
    confusion_matrices: dict[str, np.ndarray],
) -> None:
    if metrics.get("participant_count") != len(assignments):
        raise ValueError("Saved baseline participant count is inconsistent")
    if metrics.get("subject_grouped") is not True:
        raise ValueError("Saved baseline does not confirm subject-grouped evaluation")
    folds = sorted(assignments["fold"].unique().tolist())
    if len(folds) != metrics.get("folds") or folds != list(range(len(folds))):
        raise ValueError("Fold assignments do not match the saved CV design")
    for model in MODEL_ORDER:
        data = metrics.get("models", {}).get(model)
        if not data:
            raise ValueError(f"Saved baseline metrics are missing model {model}")
        aligned = predictions[model]
        truth = aligned["label"].to_numpy()
        predicted = aligned["prediction"].to_numpy()
        balanced, macro_f1, matrix, per_class = _metrics_from_predictions(
            truth, predicted
        )
        if not np.isclose(balanced, data["balanced_accuracy"], atol=1e-12):
            raise ValueError(f"{model} balanced accuracy does not match its saved OOF predictions")
        if not np.isclose(macro_f1, data["macro_f1"], atol=1e-12):
            raise ValueError(f"{model} macro-F1 does not match its saved OOF predictions")
        stored_labels = data.get("confusion_matrix_labels", [])
        stored_matrix = np.asarray(data.get("confusion_matrix", []), dtype=int)
        if stored_matrix.shape != matrix.shape:
            raise ValueError(f"{model} metrics contain an invalid confusion matrix")
        reordered = np.zeros_like(matrix)
        for row, actual in enumerate(stored_labels):
            for column, guess in enumerate(stored_labels):
                reordered[CLASS_ORDER.index(actual), CLASS_ORDER.index(guess)] = (
                    stored_matrix[row, column]
                )
        if not np.array_equal(matrix, reordered):
            raise ValueError(f"{model} saved metric confusion matrix differs from OOF predictions")
        if not np.array_equal(matrix, confusion_matrices[model]):
            raise ValueError(f"{model} confusion-matrix CSV differs from OOF predictions")
        for label in CLASS_ORDER:
            saved = data["per_class"].get(label, {})
            if (
                not np.isclose(saved.get("recall", np.nan), per_class[label]["recall"], atol=1e-12)
                or not np.isclose(saved.get("f1-score", np.nan), per_class[label]["f1"], atol=1e-12)
            ):
                raise ValueError(f"{model} saved per-class metrics differ for {label}")
        for fold_metric in data["fold_metrics"]:
            fold = int(fold_metric["fold"])
            mask = aligned["fold"].to_numpy() == fold
            fold_balanced, fold_f1, _, _ = _metrics_from_predictions(
                truth[mask], predicted[mask]
            )
            if (
                not np.isclose(fold_balanced, fold_metric["balanced_accuracy"], atol=1e-12)
                or not np.isclose(fold_f1, fold_metric["macro_f1"], atol=1e-12)
            ):
                raise ValueError(f"{model} fold {fold} metrics differ from OOF predictions")


def _read_confusion_matrix(path: Path) -> np.ndarray:
    frame = pd.read_csv(path, index_col=0)
    if set(frame.index.astype(str)) != set(CLASS_ORDER) or set(frame.columns.astype(str)) != set(CLASS_ORDER):
        raise ValueError(f"Confusion matrix has unexpected classes: {path}")
    return frame.loc[list(CLASS_ORDER), list(CLASS_ORDER)].to_numpy(dtype=int)


def _compute_group_missingness(
    features: pd.DataFrame,
    assignments: pd.DataFrame,
) -> pd.DataFrame:
    _require_columns(features, {"record_id", "subject_id", "label"}, "Subject features")
    if features["record_id"].duplicated().any():
        raise ValueError("Subject feature table contains duplicate recording IDs")
    if set(features["record_id"].astype(str)) != set(assignments["record_id"].astype(str)):
        raise ValueError("Feature dataset recording IDs differ from the approved folds")
    for key in ("subject_id", "label"):
        lookup = assignments.set_index("record_id")[key]
        values = features.set_index("record_id").loc[lookup.index, key]
        if not np.array_equal(lookup.to_numpy(), values.to_numpy()):
            raise ValueError(f"Subject features have mismatched {key} values")

    rows = []
    for modality in MODALITY_ORDER:
        columns = [column for column in features if column.startswith(f"{modality}__")]
        if not columns:
            raise ValueError(f"No {modality.upper()} features found in subject dataset")
        for label in CLASS_ORDER:
            group = features.loc[features["label"].eq(label), columns]
            rows.append({
                "label": label,
                "modality": modality,
                "subject_count": int(len(group)),
                "feature_count": int(len(columns)),
                "missing_cells": int(group.isna().sum().sum()),
                "total_cells": int(group.size),
                "missing_fraction": float(group.isna().sum().sum() / group.size),
            })
    return pd.DataFrame(rows)


def _validate_saved_missingness(recomputed: pd.DataFrame, saved_path: Path) -> pd.DataFrame:
    saved = pd.read_csv(saved_path)
    _require_columns(
        saved,
        {"label", "modality", "subject_count", "feature_count", "missing_cells", "total_cells", "missing_fraction"},
        "Saved group missingness",
    )
    merged = recomputed.merge(
        saved,
        on=["label", "modality"],
        suffixes=("_actual", "_saved"),
        validate="one_to_one",
    )
    if len(merged) != len(recomputed):
        raise ValueError("Saved missingness summary does not cover every group and modality")
    for column in ("subject_count", "feature_count", "missing_cells", "total_cells"):
        if not np.array_equal(merged[f"{column}_actual"], merged[f"{column}_saved"]):
            raise ValueError(f"Saved missingness values differ from feature data: {column}")
    if not np.allclose(
        merged["missing_fraction_actual"], merged["missing_fraction_saved"], atol=1e-12
    ):
        raise ValueError("Saved missingness percentages differ from subject feature data")
    return recomputed


def _validate_paired_outputs(
    aligned: dict[str, pd.DataFrame],
    bootstrap: pd.DataFrame,
    mcnemar: pd.DataFrame,
    changes: pd.DataFrame,
) -> None:
    pairs = (
        ("eeg_vs_eeg_eog_emg", "eeg", "eeg_eog_emg"),
        ("eeg_eog_emg_vs_eeg_eog_emg_ecg", "eeg_eog_emg", "eeg_eog_emg_ecg"),
        ("eeg_vs_eeg_eog_emg_ecg", "eeg", "eeg_eog_emg_ecg"),
    )
    for comparison, first, second in pairs:
        first_frame = aligned[first].set_index("record_id")
        second_frame = aligned[second].set_index("record_id")
        first_correct = first_frame["prediction"].eq(first_frame["label"])
        second_correct = second_frame["prediction"].eq(second_frame["label"])
        actual = changes.loc[changes["comparison"].eq(comparison)].set_index("record_id")
        if set(actual.index) != set(first_frame.index) or actual.index.duplicated().any():
            raise ValueError(f"Saved correctness changes do not cover the cohort for {comparison}")
        if not np.array_equal(
            actual.loc[first_frame.index, "first_correct"].astype(bool).to_numpy(),
            first_correct.to_numpy(),
        ) or not np.array_equal(
            actual.loc[first_frame.index, "second_correct"].astype(bool).to_numpy(),
            second_correct.to_numpy(),
        ):
            raise ValueError(f"Saved correctness changes disagree with OOF predictions for {comparison}")
        discordant_first = int((first_correct & ~second_correct).sum())
        discordant_second = int((~first_correct & second_correct).sum())
        test = mcnemar.loc[mcnemar["comparison"].eq(comparison)]
        if len(test) != 1:
            raise ValueError(f"Saved McNemar results are missing for {comparison}")
        result = test.iloc[0]
        if (
            int(result["first_correct_second_incorrect"]) != discordant_first
            or int(result["first_incorrect_second_correct"]) != discordant_second
        ):
            raise ValueError(f"Saved McNemar discordant counts disagree for {comparison}")
        discordant = discordant_first + discordant_second
        expected_p = float(
            binomtest(min(discordant_first, discordant_second), discordant, 0.5).pvalue
        ) if discordant else 1.0
        if not np.isclose(result["mcnemar_exact_two_sided_p"], expected_p, atol=1e-12):
            raise ValueError(f"Saved McNemar p-value disagrees for {comparison}")
        truth = first_frame["label"].to_numpy()
        first_ba, first_f1, _, _ = _metrics_from_predictions(
            truth, first_frame["prediction"].to_numpy()
        )
        second_ba, second_f1, _, _ = _metrics_from_predictions(
            truth, second_frame["prediction"].to_numpy()
        )
        for metric, point_difference in (
            ("balanced_accuracy", second_ba - first_ba),
            ("macro_f1", second_f1 - first_f1),
        ):
            interval = bootstrap.loc[
                bootstrap["comparison"].eq(comparison)
                & bootstrap["metric"].eq(metric)
            ]
            if len(interval) != 1 or not np.isclose(
                interval.iloc[0]["difference_second_minus_first"],
                point_difference,
                atol=1e-12,
            ):
                raise ValueError(
                    f"Saved paired-bootstrap point estimate disagrees for {comparison} {metric}"
                )
    if len(changes) != 84 * len(pairs) or changes.duplicated(
        ["comparison", "record_id"]
    ).any():
        raise ValueError("Saved paired correctness table has unexpected rows or duplicate participants")


def load_report_inputs(root: Path) -> ReportInputs:
    root = root.resolve()
    baseline_dir = root / "results" / "models" / "primary_physiology_84_v2"
    posthoc_dir = root / "results" / "analysis" / "posthoc_primary_84_v2"
    cohort_path = root / "config" / "analysis_cohorts" / "primary_physiology_84_v2.json"
    source_manifest_path = root / "config" / "cohort_manifest.json"
    feature_path = root / "results" / "features" / "physiology_primary_84_v2_candidate.csv.gz"
    preflight_path = root / "results" / "qc" / "primary_physiology_84_v2_preflight.json"
    paths = {
        "metrics": baseline_dir / "metrics.json",
        "assignments": baseline_dir / "fold_assignments.csv",
        "features": feature_path,
        "preflight": preflight_path,
        "posthoc_report": posthoc_dir / "posthoc_report.json",
        "missingness": posthoc_dir / "missingness_by_class_modality.csv",
        "harmonized_summary": posthoc_dir / "harmonized_sensitivity_summary.csv",
        "harmonized_predictions": posthoc_dir / "harmonized_sensitivity_oof_predictions.csv",
        "bootstrap": posthoc_dir / "paired_subject_bootstrap_intervals.csv",
        "mcnemar": posthoc_dir / "mcnemar_exact_tests.csv",
        "correctness_changes": posthoc_dir / "participant_correctness_changes.csv",
        "acquisition_folds": posthoc_dir / "acquisition_only_cv_by_fold.csv",
        "acquisition_associations": posthoc_dir / "acquisition_class_association.csv",
        "acquisition_metadata": posthoc_dir / "acquisition_metadata_by_record.csv",
        "cohort": cohort_path,
        "source_manifest": source_manifest_path,
    }
    for model in MODEL_ORDER:
        paths[f"predictions_{model}"] = baseline_dir / f"{model}_oof_predictions.csv"
        paths[f"confusion_{model}"] = baseline_dir / f"{model}_confusion_matrix.csv"
    missing_paths = [str(path) for path in paths.values() if not path.is_file()]
    if missing_paths:
        raise FileNotFoundError(
            "Critical saved results are missing; refusing to generate partial report: "
            + ", ".join(missing_paths)
        )

    cohort_config = _read_json(cohort_path)
    source_manifest = _read_json(source_manifest_path)
    if cohort_config.get("analysis_id") != "primary_physiology_84_v2":
        raise ValueError("The configured analysis is not the approved 84-record v2 cohort")
    included = cohort_config["included_record_ids"]
    excluded = cohort_config["excluded_records"]
    manifest_by_id = {row["record_id"]: row for row in source_manifest["records"]}
    if len(included) != len(set(included)) or len(manifest_by_id) != source_manifest["selected_record_count"]:
        raise ValueError("Cohort or original manifest contains duplicate or inconsistent records")
    if set(included) | {item["record_id"] for item in excluded} != set(manifest_by_id):
        raise ValueError("Versioned analysis cohort does not partition the original manifest")
    for row in included:
        if row not in manifest_by_id:
            raise ValueError(f"Included recording {row} is absent from source manifest")
    expected_counts = cohort_config["expected_class_counts"]
    if expected_counts != {"control": 12, "nfle": 40, "rbd": 22, "plm": 10}:
        raise ValueError("Approved cohort class counts differ from the requested cohort")

    metrics = _read_json(paths["metrics"])
    posthoc_report = _read_json(paths["posthoc_report"])
    preflight = _read_json(preflight_path)
    if metrics.get("analysis_id") != cohort_config["analysis_id"]:
        raise ValueError("Baseline metrics belong to a different cohort")
    if metrics.get("participant_count") != 84 or posthoc_report.get("participant_count") != 84:
        raise ValueError("The saved results do not contain the expected 84 recordings")
    if preflight.get("status") != "eligible" or preflight.get("error_count") != 0:
        raise ValueError("Saved cohort preflight is not eligible")
    if posthoc_report.get("status") != "complete" or posthoc_report.get("cap_annotations_used") is not False:
        raise ValueError("Post-hoc report is incomplete or used CAP annotations")
    if posthoc_report.get("baseline_outputs_modified") is not False:
        raise ValueError("Post-hoc report does not certify baseline output preservation")
    if metrics.get("reproducibility", {}).get("subject_feature_dataset_sha256") != dataset_digest(feature_path):
        raise ValueError("Subject feature dataset hash differs from the saved baseline")
    if posthoc_report.get("baseline_metrics_sha256") != dataset_digest(paths["metrics"]):
        raise ValueError("Baseline metrics hash differs from the post-hoc report")
    if posthoc_report.get("baseline_fold_assignments_sha256") != dataset_digest(paths["assignments"]):
        raise ValueError("Fold assignment hash differs from the post-hoc report")

    assignments = pd.read_csv(paths["assignments"])
    predictions = {
        model: pd.read_csv(paths[f"predictions_{model}"])
        for model in MODEL_ORDER
    }
    aligned = validate_prediction_alignment(
        assignments, predictions, set(included)
    )
    expected_labels = {record_id: manifest_by_id[record_id]["label"] for record_id in included}
    for record_id, label in expected_labels.items():
        assigned = assignments.loc[assignments["record_id"].eq(record_id), "label"].iloc[0]
        if assigned != label:
            raise ValueError(f"Fold assignment label differs from source manifest for {record_id}")
    observed_counts = assignments["label"].value_counts().to_dict()
    if observed_counts != expected_counts:
        raise ValueError(f"Saved fold class counts differ from cohort config: {observed_counts}")

    confusion_matrices = {
        model: _read_confusion_matrix(paths[f"confusion_{model}"])
        for model in MODEL_ORDER
    }
    _validate_baseline_metrics(metrics, assignments, aligned, confusion_matrices)

    features = pd.read_csv(feature_path)
    missingness_actual = _validate_saved_missingness(
        _compute_group_missingness(features, assignments), paths["missingness"]
    )
    for modality in MODALITY_ORDER:
        actual = missingness_actual.loc[missingness_actual["modality"].eq(modality)]
        expected = preflight["feature_missingness_by_modality"][modality]
        if (
            int(actual["feature_count"].iloc[0]) != int(expected["feature_count"])
            or int(actual["missing_cells"].sum()) != int(expected["missing_cells"])
            or int(actual["total_cells"].sum()) != int(expected["total_cells"])
        ):
            raise ValueError(f"Saved feature missingness differs from QC preflight for {modality}")
    harmonized_predictions = pd.read_csv(paths["harmonized_predictions"])
    _require_columns(
        harmonized_predictions,
        {"model", "record_id", "subject_id", "label", "fold", "prediction"},
        "Harmonized OOF predictions",
    )
    expected_rows = len(assignments) * len(MODEL_ORDER)
    if len(harmonized_predictions) != expected_rows or harmonized_predictions.duplicated(
        ["model", "record_id"]
    ).any():
        raise ValueError("Harmonized predictions do not contain exactly one row per model and participant")
    harmonized_summary = pd.read_csv(paths["harmonized_summary"])
    for model in MODEL_ORDER:
        rows = harmonized_predictions.loc[harmonized_predictions["model"].eq(model)]
        if set(rows["record_id"]) != set(included):
            raise ValueError(f"Harmonized predictions do not match the cohort for {model}")
        expected = assignments.set_index("record_id").loc[rows["record_id"], ["subject_id", "label", "fold"]]
        for column in ("subject_id", "label", "fold"):
            if not np.array_equal(expected[column].to_numpy(), rows[column].to_numpy()):
                raise ValueError(f"Harmonized prediction {column} mismatch for {model}")
        balanced, macro_f1, _, _ = _metrics_from_predictions(
            rows["label"].to_numpy(), rows["prediction"].to_numpy()
        )
        summary = harmonized_summary.loc[harmonized_summary["model"].eq(model)]
        if len(summary) != 1:
            raise ValueError(f"Harmonized summary is missing or duplicates {model}")
        base = metrics["models"][model]
        if (
            not np.isclose(summary.iloc[0]["balanced_accuracy"], base["balanced_accuracy"], atol=1e-12)
            or not np.isclose(summary.iloc[0]["macro_f1"], base["macro_f1"], atol=1e-12)
            or not np.isclose(summary.iloc[0]["harmonized_balanced_accuracy"], balanced, atol=1e-12)
            or not np.isclose(summary.iloc[0]["harmonized_macro_f1"], macro_f1, atol=1e-12)
        ):
            raise ValueError(f"Harmonized sensitivity summary is inconsistent for {model}")

    acquisition_folds = pd.read_csv(paths["acquisition_folds"])
    _require_columns(acquisition_folds, {"fold", "test_subject_count", "balanced_accuracy", "macro_f1", "test_class_counts"}, "Acquisition-only CV")
    if sorted(acquisition_folds["fold"].astype(int).tolist()) != sorted(assignments["fold"].unique().tolist()):
        raise ValueError("Acquisition-only folds differ from baseline fold assignments")
    for _, row in acquisition_folds.iterrows():
        fold = int(row["fold"])
        rows = assignments.loc[assignments["fold"].eq(fold)]
        if len(rows) != int(row["test_subject_count"]):
            raise ValueError(f"Acquisition-only test count differs in fold {fold}")
        if json.loads(row["test_class_counts"]) != rows["label"].value_counts().to_dict():
            raise ValueError(f"Acquisition-only class distribution differs in fold {fold}")

    acquisition_associations = pd.read_csv(paths["acquisition_associations"])
    _require_columns(acquisition_associations, {"acquisition_variable", "cramers_v"}, "Acquisition associations")
    acquisition_metadata = pd.read_csv(paths["acquisition_metadata"])
    _require_columns(acquisition_metadata, {"record_id", "subject_id", "label", "fold"}, "Acquisition metadata")
    if set(acquisition_metadata["record_id"]) != set(included):
        raise ValueError("Acquisition metadata recording IDs differ from the approved cohort")
    acquisition_indexed = acquisition_metadata.set_index("record_id").loc[
        assignments["record_id"]
    ]
    assignment_indexed = assignments.set_index("record_id").loc[
        assignments["record_id"], ["subject_id", "label", "fold"]
    ]
    for column in ("subject_id", "label", "fold"):
        if not np.array_equal(
            acquisition_indexed[column].to_numpy(),
            assignment_indexed[column].to_numpy(),
        ):
            raise ValueError(f"Acquisition metadata {column} differs from fold assignments")

    bootstrap = pd.read_csv(paths["bootstrap"])
    mcnemar = pd.read_csv(paths["mcnemar"])
    correctness_changes = pd.read_csv(paths["correctness_changes"])
    _validate_paired_outputs(
        aligned, bootstrap, mcnemar, correctness_changes
    )

    acquisition_summary = posthoc_report["acquisition_confounding"]["metadata_only_cv_summary"]["balanced_accuracy"]
    computed_mean = float(acquisition_folds["balanced_accuracy"].mean())
    computed_sd = float(acquisition_folds["balanced_accuracy"].std(ddof=1))
    if not np.isclose(computed_mean, acquisition_summary["mean_across_folds"], atol=1e-12):
        raise ValueError("Acquisition-only mean differs from its reproducibility report")
    if not np.isclose(computed_sd, acquisition_summary["standard_deviation_across_folds"], atol=1e-12):
        raise ValueError("Acquisition-only fold SD differs from its reproducibility report")

    sources = {
        key: path for key, path in paths.items()
    }
    sources["cohort_config"] = cohort_path
    sources["source_manifest"] = source_manifest_path
    return ReportInputs(
        root=root,
        assignments=assignments,
        predictions=aligned,
        metrics=metrics,
        confusion_matrices=confusion_matrices,
        harmonized_predictions=harmonized_predictions,
        harmonized_summary=harmonized_summary,
        bootstrap=bootstrap,
        mcnemar=mcnemar,
        correctness_changes=correctness_changes,
        missingness=missingness_actual,
        acquisition_folds=acquisition_folds,
        acquisition_associations=acquisition_associations,
        acquisition_metadata=acquisition_metadata,
        cohort_config=cohort_config,
        source_manifest=source_manifest,
        posthoc_report=posthoc_report,
        sources=sources,
    )


def make_report_tables(inputs: ReportInputs) -> dict[str, pd.DataFrame]:
    manifest_by_id = {row["record_id"]: row for row in inputs.source_manifest["records"]}
    included = inputs.cohort_config["included_record_ids"]
    excluded = inputs.cohort_config["excluded_records"]
    cohort_rows = []
    for label in CLASS_ORDER:
        included_count = sum(manifest_by_id[record_id]["label"] == label for record_id in included)
        excluded_rows = [
            item for item in excluded
            if manifest_by_id[item["record_id"]]["label"] == label
        ]
        exclusion_text = "; ".join(
            f'{item["record_id"]}: {item["reason"]}' for item in excluded_rows
        ) or "None"
        cohort_rows.append({
            "diagnostic_group": CLASS_LABELS[label],
            "recordings": included_count,
            "percentage_of_84_recording_cohort": 100 * included_count / len(included),
            "excluded_from_original_88_count": len(excluded_rows),
            "excluded_recordings_and_reasons": exclusion_text,
        })
    cohort_table = pd.DataFrame(cohort_rows)

    performance_rows = []
    for model in MODEL_ORDER:
        data = inputs.metrics["models"][model]
        row = {
            "model": MODEL_LABELS[model],
            "balanced_accuracy": data["balanced_accuracy"],
            "macro_f1": data["macro_f1"],
            "evaluation": "Primary baseline; pooled out-of-fold predictions",
        }
        for label in CLASS_ORDER:
            display = CLASS_LABELS[label]
            class_data = data["per_class"][label]
            row[f"{display} recall"] = class_data["recall"]
            row[f"{display} F1"] = class_data["f1-score"]
        performance_rows.append(row)
    performance_table = pd.DataFrame(performance_rows)

    pair_specs = (
        ("eeg_vs_eeg_eog_emg", "eeg", "eeg_eog_emg"),
        ("eeg_eog_emg_vs_eeg_eog_emg_ecg", "eeg_eog_emg", "eeg_eog_emg_ecg"),
        ("eeg_vs_eeg_eog_emg_ecg", "eeg", "eeg_eog_emg_ecg"),
    )
    paired_rows = []
    for comparison, first, second in pair_specs:
        ba = inputs.bootstrap.loc[
            inputs.bootstrap["comparison"].eq(comparison)
            & inputs.bootstrap["metric"].eq("balanced_accuracy")
        ]
        f1 = inputs.bootstrap.loc[
            inputs.bootstrap["comparison"].eq(comparison)
            & inputs.bootstrap["metric"].eq("macro_f1")
        ]
        mcnemar = inputs.mcnemar.loc[inputs.mcnemar["comparison"].eq(comparison)]
        changes = inputs.correctness_changes.loc[
            inputs.correctness_changes["comparison"].eq(comparison)
        ]
        if len(ba) != 1 or len(f1) != 1 or len(mcnemar) != 1 or len(changes) != 84:
            raise ValueError(f"Saved paired statistics are incomplete for {comparison}")
        corrected = int((~changes["first_correct"].astype(bool) & changes["second_correct"].astype(bool)).sum())
        regressed = int((changes["first_correct"].astype(bool) & ~changes["second_correct"].astype(bool)).sum())
        paired_rows.append({
            "models_compared": f"{MODEL_LABELS[first]} vs {MODEL_LABELS[second]}",
            "balanced_accuracy_difference_second_minus_first": ba.iloc[0]["difference_second_minus_first"],
            "balanced_accuracy_95_ci_low": ba.iloc[0]["ci_95_low"],
            "balanced_accuracy_95_ci_high": ba.iloc[0]["ci_95_high"],
            "macro_f1_difference_second_minus_first": f1.iloc[0]["difference_second_minus_first"],
            "macro_f1_95_ci_low": f1.iloc[0]["ci_95_low"],
            "macro_f1_95_ci_high": f1.iloc[0]["ci_95_high"],
            "mcnemar_exact_two_sided_p_unadjusted": mcnemar.iloc[0]["mcnemar_exact_two_sided_p"],
            "participants_corrected_by_second_model": corrected,
            "participants_regressed_by_second_model": regressed,
            "analysis": "Exploratory; paired subject bootstrap; p-values unadjusted",
        })
    paired_table = pd.DataFrame(paired_rows)

    sensitivity_rows = []
    for model in MODEL_ORDER:
        row = inputs.harmonized_summary.loc[inputs.harmonized_summary["model"].eq(model)].iloc[0]
        for metric, base_key, sensitivity_key in (
            ("Balanced accuracy", "balanced_accuracy", "harmonized_balanced_accuracy"),
            ("Macro-F1", "macro_f1", "harmonized_macro_f1"),
        ):
            base_value = float(row[base_key])
            sensitivity_value = float(row[sensitivity_key])
            difference = sensitivity_value - base_value
            sensitivity_rows.append({
                "model": MODEL_LABELS[model],
                "metric": metric,
                "original_baseline": base_value,
                "harmonized_sensitivity": sensitivity_value,
                "signed_difference_sensitivity_minus_baseline": difference,
                "absolute_difference": abs(difference),
                "analysis": "Exploratory train-fold-only feature harmonization",
            })
    sensitivity_table = pd.DataFrame(sensitivity_rows)
    return {
        "cohort_characteristics.csv": cohort_table,
        "primary_model_performance.csv": performance_table,
        "paired_statistical_comparisons.csv": paired_table,
        "sensitivity_analyses.csv": sensitivity_table,
    }


def _style() -> None:
    sns.set_theme(
        context="talk",
        style="whitegrid",
        font_scale=0.82,
        rc={
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "font.family": "DejaVu Sans",
        },
    )


def _save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for extension in ("png", "svg", "pdf"):
        path = output_dir / f"{stem}.{extension}"
        options = {"bbox_inches": "tight", "facecolor": "white"}
        if extension == "png":
            options["dpi"] = 300
        fig.savefig(path, **options)
        if extension == "svg":
            path.write_text(
                "\n".join(line.rstrip() for line in path.read_text(encoding="utf-8").splitlines())
                + "\n",
                encoding="utf-8",
            )
        paths.append(str(path))
    plt.close(fig)
    return paths


def _baseline_fold_values(inputs: ReportInputs, model: str, metric: str) -> list[float]:
    return [
        float(item[metric])
        for item in inputs.metrics["models"][model]["fold_metrics"]
    ]


def _plot_performance(inputs: ReportInputs, output_dir: Path) -> list[str]:
    fig, (axis, fold_axis) = plt.subplots(
        1, 2, figsize=(15, 6), gridspec_kw={"width_ratios": [1.1, 1]}
    )
    group_positions = np.arange(len(MODEL_ORDER))
    width = 0.31
    for index, model in enumerate(MODEL_ORDER):
        data = inputs.metrics["models"][model]
        for metric_index, (key, hatch) in enumerate((
            ("balanced_accuracy", ""),
            ("macro_f1", "///"),
        )):
            values = _baseline_fold_values(inputs, model, key)
            sd = float(np.std(values, ddof=1))
            fold_mean = float(np.mean(values))
            x = group_positions[index] + (metric_index - 0.5) * width
            value = float(data[key])
            axis.bar(
                x, value, width=width,
                color=MODEL_COLORS[model],
                edgecolor="black",
                linewidth=0.6,
                hatch=hatch,
            )
            axis.text(x, value + 0.014, f"{value:.3f}",
                      ha="center", va="bottom", fontsize=9)
            fold_axis.errorbar(
                group_positions[index] + (metric_index - 0.5) * 0.25,
                fold_mean,
                yerr=sd,
                fmt="o" if key == "balanced_accuracy" else "s",
                color=MODEL_COLORS[model],
                markersize=6,
                capsize=4,
                linewidth=1.3,
            )
    axis.set_xticks(group_positions, [MODEL_LABELS[m] for m in MODEL_ORDER], rotation=12, ha="right")
    axis.set_ylim(0, 1)
    axis.set_ylabel("Pooled out-of-fold performance")
    axis.set_title("Pooled OOF estimates")
    model_legend = [
        Patch(facecolor=MODEL_COLORS[m], edgecolor="black", label=MODEL_LABELS[m])
        for m in MODEL_ORDER
    ]
    metric_legend = [
        Patch(facecolor="white", edgecolor="black", label="Balanced accuracy"),
        Patch(facecolor="white", edgecolor="black", hatch="///", label="Macro-F1"),
    ]
    first = axis.legend(handles=model_legend, loc="upper left", fontsize=9, frameon=False)
    axis.add_artist(first)
    axis.legend(handles=metric_legend, loc="upper right", fontsize=9, frameon=False)
    fold_axis.set_xticks(group_positions, [MODEL_LABELS[m] for m in MODEL_ORDER], rotation=12, ha="right")
    fold_axis.set_ylim(0, 1)
    fold_axis.set_ylabel("Mean fold-level performance")
    fold_axis.set_title("Five-fold means ± SD")
    fold_axis.legend(
        handles=[
            Line2D([0], [0], color="black", marker="o", linestyle="None",
                   label="Balanced accuracy"),
            Line2D([0], [0], color="black", marker="s", linestyle="None",
                   label="Macro-F1"),
        ],
        loc="upper left", fontsize=9, frameon=False,
    )
    fig.suptitle("Primary model performance", fontsize=17)
    fig.text(
        0.5, 0.015,
        "Left: pooled out-of-fold estimates. Right: fold-level means with descriptive standard deviations (not confidence intervals).",
        ha="center", fontsize=9,
    )
    fig.tight_layout()
    return _save_figure(fig, output_dir, "figure_1_primary_model_performance")


def _plot_confusions(
    inputs: ReportInputs, output_dir: Path, normalized: bool = False
) -> list[str]:
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5), constrained_layout=True)
    matrices = [inputs.confusion_matrices[model] for model in MODEL_ORDER]
    maximum = max(int(matrix.max()) for matrix in matrices)
    vmax = 1.0 if normalized else maximum
    cmap = sns.color_palette("Blues", as_cmap=True)
    for axis, model, matrix in zip(axes, MODEL_ORDER, matrices, strict=True):
        row_totals = matrix.sum(axis=1, keepdims=True)
        fractions = np.divide(
            matrix, row_totals, out=np.zeros_like(matrix, dtype=float),
            where=row_totals > 0,
        )
        values = fractions if normalized else matrix
        annot = np.empty(matrix.shape, dtype=object)
        for row in range(matrix.shape[0]):
            for column in range(matrix.shape[1]):
                annot[row, column] = (
                    f"{fractions[row, column]:.0%}\n(n={matrix[row, column]})"
                    if normalized else
                    f"{matrix[row, column]}\n{fractions[row, column]:.0%}"
                )
        sns.heatmap(
            values, ax=axis, annot=annot, fmt="", cmap=cmap,
            vmin=0, vmax=vmax, cbar=(axis is axes[-1]),
            cbar_kws={"label": "Row proportion" if normalized else "Participant count"},
            linewidths=0.6, linecolor="white",
            annot_kws={"fontsize": 8},
        )
        axis.set_xticklabels([CLASS_LABELS[c] for c in CLASS_ORDER], rotation=30, ha="right")
        axis.set_yticklabels([CLASS_LABELS[c] for c in CLASS_ORDER], rotation=0)
        axis.set_xlabel("Predicted diagnosis")
        axis.set_ylabel("True diagnosis")
        axis.set_title(MODEL_LABELS[model])
    fig.suptitle(
        "Primary out-of-fold confusion matrices — row-normalized" if normalized
        else "Primary out-of-fold confusion matrices — counts and row percentages",
        fontsize=17,
    )
    stem = "figure_2b_confusion_row_normalized" if normalized else "figure_2a_confusion_counts"
    return _save_figure(fig, output_dir, stem)


def _plot_incremental_benefit(inputs: ReportInputs, output_dir: Path) -> list[str]:
    comparisons = (
        ("eeg_vs_eeg_eog_emg", "eeg", "eeg_eog_emg"),
        ("eeg_eog_emg_vs_eeg_eog_emg_ecg", "eeg_eog_emg", "eeg_eog_emg_ecg"),
    )
    metrics = ("balanced_accuracy", "macro_f1")
    rows = []
    for comparison, first, second in comparisons:
        for metric in metrics:
            result = inputs.bootstrap.loc[
                inputs.bootstrap["comparison"].eq(comparison)
                & inputs.bootstrap["metric"].eq(metric)
            ]
            if len(result) != 1:
                raise ValueError(f"Missing unique paired-bootstrap result for {comparison} {metric}")
            item = result.iloc[0]
            rows.append({
                "label": f'{metric.replace("_", " ").title()}: {MODEL_LABELS[first]} → '
                         f'{"All modalities" if second == "eeg_eog_emg_ecg" else MODEL_LABELS[second]}',
                "estimate": float(item["difference_second_minus_first"]),
                "low": float(item["ci_95_low"]),
                "high": float(item["ci_95_high"]),
                "model": second,
                "metric": metric,
            })
    fig, axis = plt.subplots(figsize=(12, 6))
    positions = np.arange(len(rows))[::-1]
    for position, row in zip(positions, rows, strict=True):
        axis.errorbar(
            row["estimate"], position,
            xerr=[[row["estimate"] - row["low"]], [row["high"] - row["estimate"]]],
            fmt="o", markersize=8, capsize=4, linewidth=1.6,
            color=MODEL_COLORS[row["model"]],
        )
        axis.text(
            row["high"] + 0.004, position,
            f'{row["estimate"]:+.3f} [{row["low"]:.3f}, {row["high"]:.3f}]',
            va="center", fontsize=9,
        )
    axis.axvline(0, color="black", linewidth=1, linestyle="--")
    axis.set_yticks(positions, [row["label"] for row in rows])
    axis.set_xlim(-0.02, 0.30)
    axis.set_xlabel("Paired difference (second model − first model)")
    axis.set_title("Exploratory incremental modality benefit")
    axis.text(
        0.5, -0.22,
        "Paired subject-bootstrap percentile 95% intervals (10,000 resamples), stratified by diagnosis; exploratory and unadjusted for multiple comparisons.",
        transform=axis.transAxes, ha="center", va="top", fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.08, 1, 0.96))
    return _save_figure(fig, output_dir, "figure_3_incremental_modality_benefit")


def _plot_sensitivity(inputs: ReportInputs, output_dir: Path) -> list[str]:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    for axis, metric, base_column, harmonized_column, title in (
        (axes[0], "balanced_accuracy", "balanced_accuracy",
         "harmonized_balanced_accuracy", "Balanced accuracy"),
        (axes[1], "macro_f1", "macro_f1",
         "harmonized_macro_f1", "Macro-F1"),
    ):
        x = np.arange(len(MODEL_ORDER))
        width = 0.34
        for index, model in enumerate(MODEL_ORDER):
            row = inputs.harmonized_summary.loc[
                inputs.harmonized_summary["model"].eq(model)
            ].iloc[0]
            for offset, key, hatch, alpha in (
                (-width / 2, base_column, "", 1.0),
                (width / 2, harmonized_column, "///", 0.58),
            ):
                value = float(row[key])
                xpos = x[index] + offset
                axis.bar(xpos, value, width, color=MODEL_COLORS[model],
                         alpha=alpha, edgecolor="black", linewidth=0.6, hatch=hatch)
                axis.text(xpos, value + 0.015, f"{value:.3f}",
                          ha="center", va="bottom", fontsize=9)
        axis.set_title(title)
        axis.set_ylim(0, 1)
        axis.set_xticks(x, [MODEL_LABELS[m] for m in MODEL_ORDER], rotation=15, ha="right")
        axis.set_ylabel("Performance (proportion)")
    axes[0].legend(
        handles=[
            Patch(facecolor="#777777", edgecolor="black", label="Primary baseline"),
            Patch(facecolor="#777777", edgecolor="black", hatch="///",
                  alpha=0.58, label="Harmonized sensitivity"),
        ],
        loc="upper left", fontsize=9, frameon=False,
    )
    fig.suptitle("Feature-selection sensitivity — exploratory analysis", fontsize=17)
    fig.text(
        0.5, -0.01,
        "Harmonization retained features with at least one observed training-fold value; test-fold values did not select features.",
        ha="center", fontsize=9,
    )
    fig.tight_layout()
    return _save_figure(fig, output_dir, "figure_4_feature_selection_sensitivity")


def _plot_acquisition(inputs: ReportInputs, output_dir: Path) -> list[str]:
    fig, (performance_axis, association_axis) = plt.subplots(
        1, 2, figsize=(16, 7), gridspec_kw={"width_ratios": [1.05, 1.2]}
    )
    acquisition = inputs.acquisition_folds.set_index("fold")["balanced_accuracy"]
    groups = [
        ("EEG", "eeg"),
        ("EEG + EMG + EOG", "eeg_eog_emg"),
        ("All modalities", "eeg_eog_emg_ecg"),
        ("Acquisition-only", None),
    ]
    names = [name for name, _ in groups]
    positions = np.arange(len(names))
    for index, (name, model) in enumerate(groups):
        if model is None:
            values = np.asarray(acquisition.to_numpy(), dtype=float)
        else:
            values = np.asarray(
                [item["balanced_accuracy"] for item in inputs.metrics["models"][model]["fold_metrics"]],
                dtype=float,
            )
        color = "#666666" if name == "Acquisition-only" else MODEL_COLORS[MODEL_ORDER[index]]
        jitter = np.linspace(-0.08, 0.08, len(values))
        performance_axis.scatter(
            np.full(len(values), index) + jitter, values, color=color,
            alpha=0.7, s=35, zorder=2,
        )
        performance_axis.errorbar(
            index, values.mean(), yerr=values.std(ddof=1), fmt="D",
            color=color, capsize=5, linewidth=1.6, markersize=7, zorder=3,
        )
        performance_axis.text(
            index, min(values.mean() + values.std(ddof=1) + 0.04, 0.98),
            f"{values.mean():.3f} ± {values.std(ddof=1):.3f}",
            ha="center", va="bottom", fontsize=8,
        )
    performance_axis.set_xticks(positions, names, rotation=10, ha="right")
    performance_axis.set_ylim(0, 1.08)
    performance_axis.set_ylabel("Fold-level balanced accuracy")
    performance_axis.set_title("Matched fold performance")
    performance_axis.text(
        0.5, -0.25, "Points = folds; diamonds and error bars = mean ± fold SD (not a CI).",
        transform=performance_axis.transAxes, ha="center", fontsize=9,
    )

    associations = inputs.acquisition_associations.copy()
    associations = associations.sort_values("cramers_v", ascending=True)
    labels = []
    for variable in associations["acquisition_variable"]:
        modality, suffix = str(variable).split("_", maxsplit=1)
        suffix = (
            suffix.replace("native_rates_hz", "native rates")
            .replace("channel_labels", "channel labels")
            .replace("channel_count", "channel count")
        )
        labels.append(f"{modality.upper()} {suffix}")
    association_axis.barh(
        np.arange(len(associations)), associations["cramers_v"],
        color="#56B4E9", edgecolor="black", linewidth=0.5,
    )
    association_axis.set_yticks(np.arange(len(associations)), labels, fontsize=9)
    association_axis.set_xlim(0, 1)
    association_axis.set_xlabel("Cramér’s V association with diagnosis")
    association_axis.set_title("Acquisition metadata association")
    for y, value in enumerate(associations["cramers_v"]):
        association_axis.text(value + 0.015, y, f"{value:.3f}", va="center", fontsize=8)
    fig.suptitle(
        "Exploratory acquisition-related diagnostic — association is not causation",
        fontsize=17,
    )
    fig.text(
        0.5, 0.005,
        "Acquisition-only classifier is diagnostic only; metadata was not used as an input to primary physiological models. Association p-values are unadjusted.",
        ha="center", fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    return _save_figure(fig, output_dir, "figure_5_acquisition_diagnostics")


def _plot_missingness(inputs: ReportInputs, output_dir: Path) -> list[str]:
    matrix = inputs.missingness.pivot(
        index="label", columns="modality", values="missing_fraction"
    ).reindex(index=CLASS_ORDER, columns=MODALITY_ORDER)
    annot = matrix.map(lambda value: f"{value:.1%}")
    fig, axis = plt.subplots(figsize=(9, 5.8))
    sns.heatmap(
        matrix * 100, annot=annot, fmt="", cmap="mako",
        vmin=0, vmax=100, linewidths=0.8, linecolor="white",
        cbar_kws={"label": "Subject-level feature cells missing (%)"},
        ax=axis, annot_kws={"fontsize": 12},
    )
    axis.set_yticklabels([CLASS_LABELS[label] for label in CLASS_ORDER], rotation=0)
    axis.set_xticklabels([MODALITY_LABELS[modality] for modality in MODALITY_ORDER], rotation=0)
    axis.set_xlabel("Physiological modality")
    axis.set_ylabel("Diagnostic group")
    axis.set_title("Missing subject-level feature values by diagnosis and modality")
    fig.text(
        0.5, -0.01,
        "Percentages are NaN feature cells among the 84 QC-eligible subject records, within each diagnostic group and modality.",
        ha="center", fontsize=9,
    )
    fig.tight_layout()
    return _save_figure(fig, output_dir, "figure_6_missing_feature_patterns")


def _caption_text(inputs: ReportInputs) -> str:
    eeg_eog_emg = inputs.harmonized_summary.loc[
        inputs.harmonized_summary["model"].eq("eeg_eog_emg")
    ].iloc[0]
    acq_mean = inputs.acquisition_folds["balanced_accuracy"].mean()
    acq_sd = inputs.acquisition_folds["balanced_accuracy"].std(ddof=1)
    eeg_assoc = inputs.acquisition_associations.loc[
        inputs.acquisition_associations["acquisition_variable"].eq("eeg_channel_labels"),
        "cramers_v",
    ]
    return f"""# Figure captions and interpretation

All primary performance estimates below use the unchanged 84-recording v2 baseline. The post-hoc results are exploratory. Each figure is exported as 300-DPI PNG, SVG, and PDF in this directory.

## Figure 1. Primary model performance

**Caption.** The left panel shows balanced accuracy and macro-F1 calculated from pooled out-of-fold predictions for the three prespecified physiological Random Forest models, with exact values above the bars. The right panel shows the mean and standard deviation of the corresponding five fold-level metric values. Fold SD describes fold-to-fold variability and is not a confidence interval.

**Interpretation.** The figure compares the primary EEG-only, EEG+EMG+EOG, and EEG+EMG+EOG+ECG models using the same participants and folds.

**Limitations.** The cohort is small and class-imbalanced. Fold SD is descriptive and should not be interpreted as sampling uncertainty or a confidence interval.

## Figure 2a. Confusion matrices: counts and row percentages

**Caption.** Each cell shows the number of participants and the percentage of that true diagnosis assigned to the predicted diagnosis. Row percentages use the true-class count as denominator; the count color scale is shared across models.

**Interpretation.** The matrices show class-specific error patterns that are hidden by aggregate metrics.

**Limitations.** Each participant contributes one out-of-fold prediction. Percentages for smaller groups are based on fewer recordings and are consequently coarse.

## Figure 2b. Row-normalized confusion matrices

**Caption.** The same out-of-fold confusion matrices are displayed as row-normalized percentages, with the corresponding count shown in each cell.

**Interpretation.** This view facilitates comparison of class-specific recall across models despite unequal class sizes.

**Limitations.** Row normalization does not change the underlying data and does not remove uncertainty from the small class counts.

## Figure 3. Incremental modality benefit

**Caption.** Paired differences in balanced accuracy and macro-F1 for the EEG-to-EEG+EMG+EOG and EEG+EMG+EOG-to-all-modality model comparisons. Points and percentile intervals come from the saved 10,000-resample paired subject bootstrap stratified by diagnosis.

**Interpretation.** The estimates quantify exploratory changes in paired out-of-fold performance after adding modality blocks.

**Limitations.** These post-hoc comparisons and intervals are exploratory; p-values/intervals are not adjusted for multiple comparisons and do not establish clinical benefit. The lower interval bounds can equal zero.

## Figure 4. Feature-selection sensitivity

**Caption.** The prespecified baseline is compared with the exploratory harmonized-feature sensitivity analysis, which selected features independently within each training fold based only on training-fold observation status. Model colors are consistent; hatching marks the sensitivity results.

**Interpretation.** For EEG+EMG+EOG, balanced accuracy changes from {eeg_eog_emg["balanced_accuracy"]:.4f} to {eeg_eog_emg["harmonized_balanced_accuracy"]:.4f}; the baseline remains the primary result.

**Limitations.** This analysis changes the feature set and should not be treated as a replacement or improvement claim for the primary experiment. No test-fold data were used to select features.

## Figure 5. Acquisition-related diagnostics

**Caption.** Left: saved fold-level balanced accuracy for each primary physiological model and the separate acquisition-only diagnostic classifier; points are folds and diamonds/error bars show mean ± fold SD. Right: saved Cramér’s V associations between acquisition metadata and diagnosis.

**Interpretation.** The acquisition-only classifier has mean fold balanced accuracy {acq_mean:.3f} (SD {acq_sd:.3f}). EEG channel-label association with diagnosis is {float(eeg_assoc.iloc[0]):.3f} by Cramér’s V.

**Limitations.** These results indicate possible acquisition-related structure, not causal confounding. The acquisition-only model is a diagnostic analysis, evaluated on the same small folds; association estimates are observational, and listed chi-square p-values are unadjusted.

## Figure 6. Missing-feature patterns

**Caption.** Heatmap of the percentage of missing subject-level feature cells by diagnostic group and modality, recomputed from the saved 84-recording subject feature table and checked against the saved post-hoc missingness table.

**Interpretation.** The figure highlights differences in feature availability across diagnostic groups and physiological modalities.

**Limitations.** A missing feature value reflects the saved feature-generation/channel coverage and does not by itself identify why a value is missing. Availability patterns should not be interpreted as physiological effects.
"""


def generate_report_figures(root: Path) -> dict:
    inputs = load_report_inputs(root)
    output_dir = inputs.root / "results" / "report_figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    _style()

    tables = make_report_tables(inputs)
    for filename, frame in tables.items():
        frame.to_csv(output_dir / filename, index=False)

    figure_outputs = {}
    figure_sources = {
        "figure_1_primary_model_performance": [
            "metrics", "assignments", "predictions_eeg",
            "predictions_eeg_eog_emg", "predictions_eeg_eog_emg_ecg",
        ],
        "figure_2a_confusion_counts": [
            "confusion_eeg", "confusion_eeg_eog_emg", "confusion_eeg_eog_emg_ecg",
        ],
        "figure_2b_confusion_row_normalized": [
            "confusion_eeg", "confusion_eeg_eog_emg", "confusion_eeg_eog_emg_ecg",
        ],
        "figure_3_incremental_modality_benefit": ["bootstrap"],
        "figure_4_feature_selection_sensitivity": [
            "metrics", "harmonized_summary", "harmonized_predictions",
        ],
        "figure_5_acquisition_diagnostics": [
            "metrics", "acquisition_folds", "acquisition_associations",
            "acquisition_metadata", "posthoc_report",
        ],
        "figure_6_missing_feature_patterns": [
            "features", "missingness", "preflight", "cohort_config",
        ],
    }
    for key, function in (
        ("figure_1_primary_model_performance", _plot_performance),
        ("figure_2a_confusion_counts", lambda data, out: _plot_confusions(data, out, False)),
        ("figure_2b_confusion_row_normalized", lambda data, out: _plot_confusions(data, out, True)),
        ("figure_3_incremental_modality_benefit", _plot_incremental_benefit),
        ("figure_4_feature_selection_sensitivity", _plot_sensitivity),
        ("figure_5_acquisition_diagnostics", _plot_acquisition),
        ("figure_6_missing_feature_patterns", _plot_missingness),
    ):
        figure_outputs[key] = function(inputs, output_dir)

    excel_status = "not generated: neither openpyxl nor xlsxwriter is installed"
    try:
        import importlib.util
        if importlib.util.find_spec("openpyxl") or importlib.util.find_spec("xlsxwriter"):
            with pd.ExcelWriter(output_dir / "report_tables.xlsx") as writer:
                for filename, frame in tables.items():
                    frame.to_excel(writer, sheet_name=filename.removesuffix(".csv")[:31], index=False)
            excel_status = "generated"
    except (ImportError, ModuleNotFoundError) as error:
        excel_status = f"not generated: {error}"

    source_hashes = {
        key: {
            "path": str(path.relative_to(inputs.root)),
            "sha256": dataset_digest(path),
        }
        for key, path in inputs.sources.items()
    }
    manifest = {
        "analysis": "primary_physiology_84_v2_report_figures",
        "status": "complete",
        "primary_results_modified": False,
        "models_retrained": False,
        "feature_extraction_rerun": False,
        "cohort_count": len(inputs.assignments),
        "class_counts": inputs.assignments["label"].value_counts().to_dict(),
        "figure_export_formats": {"png": "300 DPI", "svg": "editable vector", "pdf": "publication insertion"},
        "figures": {
            key: {
                "outputs": [str(Path(path).relative_to(inputs.root)) for path in figure_outputs[key]],
                "source_files": [source_hashes[source] for source in figure_sources[key]],
            }
            for key in figure_outputs
        },
        "tables": {
            filename: {
                "path": filename,
                "source_files": [
                    source_hashes[source]
                    for source in (
                        ["metrics", "assignments", "cohort_config", "source_manifest"]
                        if filename in ("cohort_characteristics.csv", "primary_model_performance.csv")
                        else ["bootstrap", "mcnemar", "correctness_changes"]
                        if filename == "paired_statistical_comparisons.csv"
                        else ["metrics", "harmonized_summary", "harmonized_predictions"]
                    )
                ],
            }
            for filename in tables
        },
        "xlsx_export": excel_status,
        "source_hashes": source_hashes,
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "matplotlib": matplotlib.__version__,
            "seaborn": sns.__version__,
        },
    }
    (output_dir / "figure_manifest.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    (output_dir / "figure_captions.md").write_text(
        _caption_text(inputs), encoding="utf-8"
    )
    manifest["outputs"] = [
        str(path.relative_to(inputs.root))
        for path in output_dir.iterdir()
        if path.is_file()
    ]
    (output_dir / "report_summary.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return {
        "status": "complete",
        "output_directory": str(output_dir),
        "participant_count": len(inputs.assignments),
        "figures_generated": list(figure_outputs),
        "tables_generated": list(tables),
        "xlsx_export": excel_status,
        "baseline_results_modified": False,
    }
