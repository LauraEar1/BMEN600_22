import numpy as np
import pandas as pd
import pytest

from capsleep.posthoc import (
    align_baseline_predictions,
    calculate_missingness,
    exact_mcnemar,
    paired_correctness_changes,
    paired_stratified_bootstrap,
)


def _aligned_inputs():
    assignments = pd.DataFrame({
        "record_id": ["s1", "s2", "s3", "s4"],
        "subject_id": ["s1", "s2", "s3", "s4"],
        "label": ["a", "a", "b", "b"],
        "fold": [0, 1, 0, 1],
    })
    predictions = {}
    values = {
        "eeg": ["a", "b", "b", "a"],
        "eeg_eog_emg": ["a", "a", "a", "b"],
        "eeg_eog_emg_ecg": ["b", "a", "b", "b"],
    }
    for model, predicted in values.items():
        predictions[model] = assignments.assign(prediction=predicted)
    return assignments, predictions


def test_alignment_requires_identical_recordings_labels_and_fold_assignments():
    assignments, predictions = _aligned_inputs()

    aligned = align_baseline_predictions(assignments, predictions)

    assert aligned["record_id"].tolist() == assignments["record_id"].tolist()
    assert aligned["eeg_correct"].tolist() == [True, False, True, False]
    broken = {key: value.copy() for key, value in predictions.items()}
    broken["eeg_eog_emg"].loc[0, "fold"] = 1
    with pytest.raises(ValueError, match="fold differs"):
        align_baseline_predictions(assignments, broken)


def test_mcnemar_and_participant_correctness_changes_are_paired():
    assignments, predictions = _aligned_inputs()
    aligned = align_baseline_predictions(assignments, predictions)

    changes = paired_correctness_changes(aligned)
    comparison = changes[changes["comparison"].eq("eeg_vs_eeg_eog_emg")]
    assert len(comparison) == 4
    assert comparison["change"].tolist() == [
        "both_correct", "eeg_eog_emg_improved",
        "eeg_eog_emg_regressed", "eeg_eog_emg_improved",
    ]
    tests = exact_mcnemar(aligned)
    row = tests.loc[tests["comparison"].eq("eeg_vs_eeg_eog_emg")].iloc[0]
    assert row["first_correct_second_incorrect"] == 1
    assert row["first_incorrect_second_correct"] == 2
    assert row["mcnemar_exact_two_sided_p"] == pytest.approx(1.0)


def test_paired_stratified_bootstrap_is_reproducible_and_paired():
    assignments, predictions = _aligned_inputs()
    aligned = align_baseline_predictions(assignments, predictions)
    first = paired_stratified_bootstrap(
        aligned, "eeg", "eeg_eog_emg", ["a", "b"],
        repetitions=200, random_seed=11,
    )
    second = paired_stratified_bootstrap(
        aligned, "eeg", "eeg_eog_emg", ["a", "b"],
        repetitions=200, random_seed=11,
    )

    assert first == second
    assert first["metrics"]["balanced_accuracy"]["ci_95_percentile"][0] <= (
        first["metrics"]["balanced_accuracy"]["ci_95_percentile"][1]
    )


def test_missingness_is_reported_per_record_and_diagnostic_group():
    frame = pd.DataFrame({
        "record_id": ["a1", "a2", "b1"],
        "subject_id": ["a1", "a2", "b1"],
        "label": ["a", "a", "b"],
        "eeg__x": [1.0, np.nan, 3.0],
        "emg__x": [np.nan, 2.0, 4.0],
    })
    record, group = calculate_missingness(
        frame,
        {"eeg": ["eeg__x"], "emg": ["emg__x"]},
    )

    a_eeg = record.query("record_id == 'a' and modality == 'eeg'")
    assert a_eeg.empty
    assert record.query("record_id == 'a2' and modality == 'eeg'").iloc[0][
        "missing_fraction"
    ] == 1.0
    a_group = group.query("label == 'a' and modality == 'eeg'").iloc[0]
    assert a_group["missing_cells"] == 1
    assert a_group["total_cells"] == 2
    assert a_group["missing_fraction"] == 0.5
