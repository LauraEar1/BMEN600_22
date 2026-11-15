from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from capsleep.report_figures import (
    CLASS_ORDER,
    MODEL_ORDER,
    _compute_group_missingness,
    _metrics_from_predictions,
    _plot_confusions,
    validate_prediction_alignment,
)


def _assignments_and_predictions():
    assignments = pd.DataFrame({
        "record_id": ["r1", "r2", "r3", "r4"],
        "subject_id": ["r1", "r2", "r3", "r4"],
        "label": ["control", "control", "nfle", "nfle"],
        "fold": [0, 1, 0, 1],
    })
    predictions = {
        model: assignments.assign(
            prediction=["control", "nfle", "nfle", "control"]
        )
        for model in MODEL_ORDER
    }
    return assignments, predictions


def test_prediction_alignment_rejects_mismatched_labels_folds_and_ids():
    assignments, predictions = _assignments_and_predictions()
    aligned = validate_prediction_alignment(
        assignments, predictions, set(assignments["record_id"])
    )
    assert aligned["eeg"]["record_id"].tolist() == assignments["record_id"].tolist()

    changed = {key: value.copy() for key, value in predictions.items()}
    changed["eeg_eog_emg"].loc[0, "label"] = "nfle"
    with pytest.raises(ValueError, match="label differs"):
        validate_prediction_alignment(assignments, changed)

    changed = {key: value.copy() for key, value in predictions.items()}
    changed["eeg_eog_emg_ecg"].loc[0, "fold"] = 1
    with pytest.raises(ValueError, match="fold differs"):
        validate_prediction_alignment(assignments, changed)

    with pytest.raises(ValueError, match="approved cohort IDs"):
        validate_prediction_alignment(assignments, predictions, {"r1", "r2"})


def test_saved_metric_calculation_returns_confusion_and_per_class_metrics():
    truth = np.array(["control", "control", "nfle", "nfle"])
    predicted = np.array(["control", "nfle", "nfle", "control"])

    balanced, macro_f1, matrix, per_class = _metrics_from_predictions(
        truth, predicted, labels=("control", "nfle")
    )

    assert matrix.tolist() == [[1, 1], [1, 1]]
    assert balanced == pytest.approx(0.5)
    assert macro_f1 == pytest.approx(0.5)
    assert per_class["control"]["recall"] == pytest.approx(0.5)


def test_missing_feature_percentages_are_grouped_from_actual_values():
    assignments = pd.DataFrame({
        "record_id": ["c1", "c2", "n1", "r1", "p1"],
        "subject_id": ["c1", "c2", "n1", "r1", "p1"],
        "label": ["control", "control", "nfle", "rbd", "plm"],
        "fold": [0, 1, 0, 1, 0],
    })
    features = assignments[["record_id", "subject_id", "label"]].copy()
    features["eeg__x"] = [1.0, np.nan, 2.0, 3.0, 4.0]
    features["emg__x"] = [np.nan, np.nan, 4.0, 5.0, 6.0]
    features["eog__x"] = [1.0, 2.0, 3.0, np.nan, 5.0]
    features["ecg__x"] = [1.0, 2.0, 3.0, 4.0, np.nan]

    summary = _compute_group_missingness(features, assignments)
    healthy_emg = summary.query(
        "label == 'control' and modality == 'emg'"
    ).iloc[0]
    assert healthy_emg["missing_fraction"] == pytest.approx(1.0)
    assert healthy_emg["subject_count"] == 2


def test_confusion_figure_exports_png_svg_and_pdf(tmp_path):
    matrices = {
        model: np.eye(len(CLASS_ORDER), dtype=int) * 2
        for model in MODEL_ORDER
    }
    inputs = SimpleNamespace(confusion_matrices=matrices)

    outputs = _plot_confusions(inputs, tmp_path, normalized=True)

    assert {path.rsplit(".", 1)[1] for path in outputs} == {"png", "svg", "pdf"}
    assert all((tmp_path / f"figure_2b_confusion_row_normalized.{suffix}").is_file()
               for suffix in ("png", "svg", "pdf"))
