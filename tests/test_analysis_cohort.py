from dataclasses import replace
import json
from pathlib import Path

import pytest

from capsleep.analysis_cohort import _modality_availability, train_analysis_cohort
from capsleep.config import load_settings


def test_analysis_cohort_keeps_original_manifest_exclusions_explicit():
    config = json.loads(
        Path("config/analysis_cohorts/primary_physiology_85_v1.json").read_text(
            encoding="utf-8"
        )
    )
    included = set(config["included_record_ids"])
    excluded = {row["record_id"]: row for row in config["excluded_records"]}

    assert len(included) == 85
    assert set(config["expected_class_counts"].values()) == {10, 13, 22, 40}
    assert set(excluded) == {"n13", "n14", "n15"}
    assert all("blank physical-dimension fields" in row["reason"] for row in excluded.values())
    assert config["eligibility"]["required_modalities"] == ["eeg", "emg", "eog", "ecg"]


def test_missing_or_qc_failed_modalities_block_a_recording():
    channels = [
        {
            "modality": "eeg",
            "excluded": False,
            "usable_window_count": 10,
            "physical_unit": "uV",
        },
        {
            "modality": "emg",
            "excluded": True,
            "usable_window_count": 0,
            "physical_unit": "uV",
        },
    ]

    missing, unusable = _modality_availability(
        channels, {"eeg", "emg", "eog", "ecg"}
    )

    assert missing == ["ecg", "eog"]
    assert unusable == ["ecg", "emg", "eog"]


def test_training_refuses_an_ineligible_analysis_cohort(tmp_path):
    settings = replace(load_settings(), root=tmp_path)
    analysis_id = "primary_physiology_84_v2"
    report_path = tmp_path / "results" / "qc" / f"{analysis_id}_preflight.json"
    report_path.parent.mkdir(parents=True)
    report_path.write_text(
        json.dumps({"analysis_id": analysis_id, "status": "blocked"}),
        encoding="utf-8",
    )
    config_path = tmp_path / "cohort.json"
    config_path.write_text(
        json.dumps({"analysis_id": analysis_id}), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="is not eligible"):
        train_analysis_cohort(settings, config_path)

    assert not (tmp_path / "results" / "models" / analysis_id).exists()
