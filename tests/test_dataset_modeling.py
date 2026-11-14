import numpy as np
import pandas as pd

from capsleep.dataset import feature_columns, validate_modality_datasets
from capsleep.modeling import evaluate_cap_models, evaluate_primary_models, make_splits


def _cohort():
    labels = ["control"] * 16 + ["nfle"] * 40 + ["rbd"] * 22 + ["plm"] * 10
    frame = pd.DataFrame({
        "record_id": [f"subject{index}" for index in range(88)],
        "subject_id": [f"subject{index}" for index in range(88)],
        "label": labels,
    })
    rng = np.random.default_rng(42)
    for modality in ("eeg", "eog", "emg", "ecg"):
        frame[f"{modality}__feature"] = rng.normal(size=88)
    frame["cap_rate_percent"] = rng.normal(size=88)
    return frame


def test_primary_modality_sets_do_not_include_annotations():
    frame = _cohort()
    sets = validate_modality_datasets(frame)
    assert all(not any("cap" in column for column in columns) for columns in sets.values())
    assert len(sets["eeg"]) == 1
    assert len(sets["eeg_eog_emg"]) == 3
    assert len(sets["eeg_eog_emg_ecg"]) == 4
    assert feature_columns(frame, ("eeg__",)) == ["eeg__feature"]


def test_group_splits_partition_subjects_without_overlap():
    frame = _cohort()
    splits = make_splits(frame, folds=5, random_seed=42)
    assert len(splits) == 5
    assert all(
        not set(frame.iloc[train]["subject_id"]) & set(frame.iloc[test]["subject_id"])
        for train, test in splits
    )


def test_models_share_fold_assignments_and_metrics(tmp_path):
    frame = _cohort()
    result = evaluate_primary_models(frame, tmp_path, folds=5, random_seed=42, trees=5)
    folds = pd.read_csv(tmp_path / "fold_assignments.csv")
    for name in ("eeg", "eeg_eog_emg", "eeg_eog_emg_ecg"):
        predictions = pd.read_csv(tmp_path / f"{name}_oof_predictions.csv")
        assert predictions["fold"].tolist() == folds["fold"].tolist()
        assert "balanced_accuracy" in result["models"][name]
        assert len(result["models"][name]["confusion_matrix"]) == 4


def test_84_record_cohort_uses_configured_counts_and_reports_fold_metrics(tmp_path):
    frame = _cohort().iloc[list(range(12)) + list(range(16, 88))].reset_index(drop=True)
    class_counts = {"control": 12, "nfle": 40, "rbd": 22, "plm": 10}

    result = evaluate_primary_models(
        frame,
        tmp_path,
        folds=5,
        random_seed=42,
        trees=2,
        expected_class_counts=class_counts,
        analysis_id="primary_physiology_84_v2",
    )

    folds = pd.read_csv(tmp_path / "fold_assignments.csv")
    assert len(folds) == 84
    assert result["analysis_id"] == "primary_physiology_84_v2"
    for model_name in ("eeg", "eeg_eog_emg", "eeg_eog_emg_ecg"):
        predictions = pd.read_csv(tmp_path / f"{model_name}_oof_predictions.csv")
        assert predictions["fold"].tolist() == folds["fold"].tolist()
        assert predictions["subject_id"].tolist() == folds["subject_id"].tolist()
        assert len(result["models"][model_name]["fold_metrics"]) == 5
        assert all(
            "precision" in metrics and "recall" in metrics
            for fold in result["models"][model_name]["fold_metrics"]
            for metrics in fold["per_class"].values()
        )


def test_cap_experiment_compares_eeg_baseline_and_addition_on_same_folds(tmp_path):
    frame = _cohort()
    physiology = frame.drop(columns=["cap_rate_percent"])
    cap = frame[["record_id", "subject_id", "label", "cap_rate_percent"]]
    primary = evaluate_primary_models(
        physiology, tmp_path / "primary", folds=5, random_seed=42, trees=2
    )
    secondary = evaluate_cap_models(
        physiology, cap, ["cap_rate_percent"], tmp_path / "cap",
        folds=5, random_seed=42, trees=2,
    )
    primary_folds = pd.read_csv(tmp_path / "primary" / "fold_assignments.csv")
    cap_folds = pd.read_csv(tmp_path / "cap" / "fold_assignments.csv")
    assert primary_folds["fold"].tolist() == cap_folds["fold"].tolist()
    assert set(secondary["models"]) == {"eeg_baseline", "eeg_plus_cap"}
    assert secondary["models"]["eeg_plus_cap"]["feature_count"] == 2
