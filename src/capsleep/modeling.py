from __future__ import annotations

from pathlib import Path
import json
import platform

import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline

from .dataset import (
    EXPECTED_CLASS_COUNTS,
    feature_columns,
    validate_modality_datasets,
)


def make_splits(
    frame: pd.DataFrame, folds: int = 5, random_seed: int = 42
) -> list[tuple[np.ndarray, np.ndarray]]:
    if "subject_id" not in frame:
        raise ValueError("Subject identifiers are required for grouped cross-validation")
    splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=random_seed)
    splits = list(splitter.split(frame, frame["label"], groups=frame["subject_id"]))
    validation = np.concatenate([test for _, test in splits])
    if sorted(validation.tolist()) != list(range(len(frame))):
        raise ValueError("Cross-validation folds do not partition the cohort exactly once")
    for train, test in splits:
        if set(frame.iloc[train]["subject_id"]) & set(frame.iloc[test]["subject_id"]):
            raise ValueError("Subject leakage between training and validation folds")
        if set(frame.iloc[test]["label"]) != set(frame["label"]):
            raise ValueError("A validation fold is missing one or more diagnostic classes")
    return splits


def evaluate_primary_models(
    frame: pd.DataFrame,
    output_directory: Path,
    folds: int = 5,
    random_seed: int = 42,
    trees: int = 500,
    expected_class_counts: dict[str, int] = EXPECTED_CLASS_COUNTS,
    analysis_id: str = "primary_88",
) -> dict:
    datasets = validate_modality_datasets(frame, expected_class_counts)
    labels = sorted(frame["label"].unique().tolist())
    splits = make_splits(frame, folds, random_seed)
    assignments = np.full(len(frame), -1, dtype=int)
    for fold, (_, test) in enumerate(splits):
        assignments[test] = fold
    output_directory.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "record_id": frame["record_id"],
        "subject_id": frame["subject_id"],
        "label": frame["label"],
        "fold": assignments,
    }).to_csv(output_directory / "fold_assignments.csv", index=False)
    results = {}
    for model_name, columns in datasets.items():
        predictions = np.empty(len(frame), dtype=object)
        fold_metrics = []
        for fold_index, (train, test) in enumerate(splits):
            estimator = make_pipeline(
                SimpleImputer(strategy="median", keep_empty_features=True),
                RandomForestClassifier(
                    n_estimators=trees,
                    class_weight="balanced_subsample",
                    random_state=random_seed,
                    n_jobs=-1,
                ),
            )
            estimator.fit(frame.iloc[train][columns], frame.iloc[train]["label"])
            fold_predictions = estimator.predict(frame.iloc[test][columns])
            predictions[test] = fold_predictions
            fold_truth = frame.iloc[test]["label"].to_numpy()
            fold_matrix = confusion_matrix(
                fold_truth, fold_predictions, labels=labels
            )
            fold_report = classification_report(
                fold_truth,
                fold_predictions,
                labels=labels,
                output_dict=True,
                zero_division=0,
            )
            fold_metrics.append({
                "fold": fold_index,
                "test_subject_count": len(test),
                "test_class_counts": frame.iloc[test]["label"].value_counts().to_dict(),
                "training_features_all_missing": int(
                    frame.iloc[train][columns].isna().all(axis=0).sum()
                ),
                "balanced_accuracy": float(
                    balanced_accuracy_score(fold_truth, fold_predictions)
                ),
                "macro_f1": float(
                    f1_score(
                        fold_truth, fold_predictions, labels=labels, average="macro"
                    )
                ),
                "per_class": {
                    label: fold_report[label] for label in labels
                },
                "confusion_matrix": fold_matrix.tolist(),
            })
        truth = frame["label"].to_numpy()
        matrix = confusion_matrix(truth, predictions, labels=labels)
        report = classification_report(
            truth, predictions, labels=labels, output_dict=True, zero_division=0
        )
        results[model_name] = {
            "feature_count": len(columns),
            "feature_columns": columns,
            "balanced_accuracy": float(balanced_accuracy_score(truth, predictions)),
            "macro_f1": float(f1_score(truth, predictions, labels=labels, average="macro")),
            "accuracy": float(accuracy_score(truth, predictions)),
            "per_class": {label: report[label] for label in labels},
            "confusion_matrix": matrix.tolist(),
            "confusion_matrix_labels": labels,
            "fold_metrics": fold_metrics,
        }
        pd.DataFrame({
            "record_id": frame["record_id"],
            "subject_id": frame["subject_id"],
            "label": truth,
            "prediction": predictions,
            "fold": assignments,
        }).to_csv(output_directory / f"{model_name}_oof_predictions.csv", index=False)
        pd.DataFrame(matrix, index=labels, columns=labels).to_csv(
            output_directory / f"{model_name}_confusion_matrix.csv"
        )
    report = {
        "experiment": "primary_physiology_only",
        "analysis_id": analysis_id,
        "participant_count": len(frame),
        "class_counts": frame["label"].value_counts().to_dict(),
        "folds": folds,
        "random_seed": random_seed,
        "subject_grouped": True,
        "imputer": "median, fit within each training fold",
        "feature_selection": "none",
        "hyperparameter_tuning": "none",
        "evaluation_unit": "one subject-level row per recording",
        "classifier": "RandomForestClassifier",
        "n_estimators": trees,
        "class_weight": "balanced_subsample",
        "n_jobs": -1,
        "models": results,
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
        },
    }
    (output_directory / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def evaluate_cap_models(
    physiology: pd.DataFrame,
    cap: pd.DataFrame,
    cap_columns: list[str],
    output_directory: Path,
    folds: int = 5,
    random_seed: int = 42,
    trees: int = 500,
) -> dict:
    identifiers = ["record_id", "subject_id", "label"]
    if not set(identifiers).issubset(physiology.columns) or not set(identifiers).issubset(cap.columns):
        raise ValueError("CAP-informed experiment requires identifiers and labels in both datasets")
    if physiology[identifiers].to_dict("records") != cap[identifiers].to_dict("records"):
        raise ValueError("CAP and physiology datasets must contain identical ordered participants and labels")
    if len(physiology) != 88 or physiology["subject_id"].duplicated().any():
        raise ValueError("CAP-informed experiment requires all 88 unique participants")
    if physiology["label"].value_counts().to_dict() != EXPECTED_CLASS_COUNTS:
        raise ValueError("CAP-informed experiment requires the verified four-class distribution")
    if not cap_columns or not set(cap_columns).issubset(cap.columns):
        raise ValueError("CAP-informed experiment requires explicit annotation feature columns")
    eeg_columns = feature_columns(physiology, ("eeg__",))
    if not eeg_columns:
        raise ValueError("CAP-informed experiment requires EEG physiology features")
    frame = pd.concat(
        [physiology[identifiers + eeg_columns].reset_index(drop=True),
         cap[cap_columns].reset_index(drop=True)],
        axis=1,
    )
    if frame[eeg_columns + cap_columns].isna().all(axis=0).any():
        raise ValueError("CAP-informed data contain a feature that is entirely missing")
    splits = make_splits(frame, folds, random_seed)
    labels = sorted(frame["label"].unique().tolist())
    output_directory.mkdir(parents=True, exist_ok=True)
    fold_assignments = np.full(len(frame), -1, dtype=int)
    for fold, (_, test) in enumerate(splits):
        fold_assignments[test] = fold
    pd.DataFrame({
        "record_id": frame["record_id"],
        "subject_id": frame["subject_id"],
        "label": frame["label"],
        "fold": fold_assignments,
    }).to_csv(output_directory / "fold_assignments.csv", index=False)
    model_sets = {
        "eeg_baseline": eeg_columns,
        "eeg_plus_cap": eeg_columns + cap_columns,
    }
    results = {}
    for name, columns in model_sets.items():
        predictions = np.empty(len(frame), dtype=object)
        for train, test in splits:
            model = make_pipeline(
                SimpleImputer(strategy="median", keep_empty_features=True),
                RandomForestClassifier(
                    n_estimators=trees, class_weight="balanced_subsample",
                    random_state=random_seed, n_jobs=-1,
                ),
            )
            model.fit(frame.iloc[train][columns], frame.iloc[train]["label"])
            predictions[test] = model.predict(frame.iloc[test][columns])
        matrix = confusion_matrix(frame["label"], predictions, labels=labels)
        report = classification_report(
            frame["label"], predictions, labels=labels, output_dict=True, zero_division=0
        )
        results[name] = {
            "feature_count": len(columns),
            "balanced_accuracy": float(balanced_accuracy_score(frame["label"], predictions)),
            "macro_f1": float(f1_score(frame["label"], predictions, average="macro")),
            "per_class": {label: report[label] for label in labels},
            "confusion_matrix": matrix.tolist(),
            "confusion_matrix_labels": labels,
        }
        pd.DataFrame({
            "record_id": frame["record_id"],
            "label": frame["label"],
            "prediction": predictions,
            "fold": fold_assignments,
        }).to_csv(output_directory / f"{name}_oof_predictions.csv", index=False)
        pd.DataFrame(matrix, index=labels, columns=labels).to_csv(
            output_directory / f"{name}_confusion_matrix.csv"
        )
    result = {
        "experiment": "secondary_eeg_plus_cap_vs_eeg_baseline",
        "participant_count": len(frame),
        "folds": folds,
        "random_seed": random_seed,
        "subject_grouped": True,
        "imputer": "median, fit within each training fold",
        "models": results,
    }
    (output_directory / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    return result
