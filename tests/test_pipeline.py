from dataclasses import replace
import json
from pathlib import Path

import pandas as pd
import pytest

from capsleep import pipeline
from capsleep.cohort import load_manifest
from capsleep.config import load_settings
from capsleep.features import feature_schema


def test_physiology_extraction_does_not_depend_on_annotation_validation(
    tmp_path, monkeypatch
):
    records = [
        {"record_id": "sample1", "filename": "sample1.edf", "label": "control"},
        {"record_id": "sample2", "filename": "sample2.edf", "label": "control"},
    ]
    monkeypatch.setattr(
        pipeline,
        "verify_manifest",
        lambda path: {"status": "verified", "record_count": len(records)},
    )
    monkeypatch.setattr(
        pipeline,
        "load_manifest",
        lambda path: {"inventory_sha256": "test-manifest", "records": records},
    )
    monkeypatch.setattr(
        pipeline.requests,
        "get",
        lambda *args, **kwargs: pytest.fail("physiology extraction requested CAP annotations"),
    )
    downloaded = []

    def fake_download(url, temporary_directory, minimum_free_gb, timeout):
        record_id = Path(url).stem
        path = temporary_directory / f".capsleep_{record_id}_test.edf"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"synthetic")
        downloaded.append(path)
        return path

    def fake_extract(path, record_id, label, settings):
        row = {
            "record_id": record_id,
            "subject_id": record_id,
            "label": label,
            "eeg__F3_C3__rms_median": 2.0,
        }
        qc = {
            "record_id": record_id,
            "label": label,
            "duration_seconds": 30,
            "channel_count": 1,
            "recognized_channel_count": 1,
            "excluded_channel_count": 0,
            "missing_modalities": "emg|eog|ecg",
            "qc_excluded_channels": "",
            "status": "complete",
        }
        channels = [{
            "record_id": record_id,
            "channel": "F3-C3",
            "modality": "eeg",
            "physical_unit": "uV",
            "native_sfreq_hz": 128.0,
            "nonfinite_fraction": 0.0,
            "flat_window_fraction": 0.0,
            "saturated_fraction": 0.0,
            "usable_window_count": 1,
            "excluded": False,
            "exclusion_reasons": "",
        }]
        windows = [{
            "record_id": record_id,
            "subject_id": record_id,
            "label": label,
            "window_index": 0,
            "window_start_seconds": 0,
            "channel": "F3-C3",
            "modality": "eeg",
            "physical_unit": "uV",
            "native_sfreq_hz": 128.0,
            "qc_is_flat": False,
            "qc_nonfinite_fraction": 0.0,
            "qc_saturated_fraction": 0.0,
            "qc_channel_excluded": False,
            "feature__rms": 2.0,
        }]
        return row, qc, channels, windows

    monkeypatch.setattr(pipeline, "download_recording", fake_download)
    monkeypatch.setattr(pipeline, "extract_recording", fake_extract)
    settings = replace(
        load_settings(),
        root=tmp_path,
        manifest_path=tmp_path / "config" / "cohort_manifest.json",
        expected_records=len(records),
    )

    result = pipeline.extract_cohort(settings)

    assert result["physiology_rows"] == 2
    assert result["annotation_validation"].startswith("independent")
    assert all(not path.exists() for path in downloaded)
    assert (tmp_path / "results" / "features" / "physiology_features.csv.gz").exists()
    assert (tmp_path / "results" / "features" / "physiology_window_features.csv.gz").exists()
    assert not (tmp_path / "results" / "features" / "cap_features.csv.gz").exists()
    assert not (tmp_path / "results" / "cap_validation.json").exists()
    frame = pd.read_csv(tmp_path / "results" / "features" / "physiology_features.csv.gz")
    assert frame["record_id"].tolist() == ["sample1", "sample2"]


def test_extraction_logs_qc_failure_and_attempts_remaining_records(tmp_path, monkeypatch):
    records = [
        {"record_id": "sample1", "filename": "sample1.edf", "label": "control"},
        {"record_id": "sample2", "filename": "sample2.edf", "label": "control"},
    ]
    monkeypatch.setattr(
        pipeline,
        "verify_manifest",
        lambda path: {"status": "verified", "record_count": len(records)},
    )
    monkeypatch.setattr(
        pipeline,
        "load_manifest",
        lambda path: {"inventory_sha256": "test-manifest", "records": records},
    )
    downloaded = []

    def fake_download(url, temporary_directory, minimum_free_gb, timeout):
        record_id = Path(url).stem
        path = temporary_directory / f".capsleep_{record_id}_test.edf"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"synthetic")
        downloaded.append(path)
        return path

    def fake_extract(path, record_id, label, settings):
        status = "failed" if record_id == "sample1" else "complete"
        return (
            {"record_id": record_id, "subject_id": record_id, "label": label},
            {
                "record_id": record_id,
                "status": status,
                "failure_reason": "no usable EEG" if status == "failed" else "",
            },
            [{
                "record_id": record_id,
                "channel": "F3-C3",
                "feature_channel": "F3-C3",
                "modality": "eeg",
                "physical_unit": "",
                "excluded": status == "failed",
                "exclusion_reasons": "missing_or_unsupported_physical_unit",
            }],
            [],
        )

    monkeypatch.setattr(pipeline, "download_recording", fake_download)
    monkeypatch.setattr(pipeline, "extract_recording", fake_extract)
    settings = replace(
        load_settings(),
        root=tmp_path,
        manifest_path=tmp_path / "config" / "cohort_manifest.json",
        expected_records=len(records),
    )

    with pytest.raises(RuntimeError, match="no final feature dataset was written"):
        pipeline.extract_cohort(settings)

    assert len(downloaded) == 2
    assert downloaded[0].exists()
    assert not downloaded[1].exists()
    checkpoint = json.loads(
        (tmp_path / "results" / "checkpoints" / "extraction_checkpoint.json").read_text()
    )
    assert checkpoint["records"]["sample1"]["status"] == "failed"
    assert checkpoint["records"]["sample2"]["status"] == "complete"
    qc_path = Path(checkpoint["records"]["sample1"]["qc_summary"])
    assert qc_path.exists()
    assert not (tmp_path / "results" / "features" / "physiology_features.csv.gz").exists()
    failures = pd.read_csv(tmp_path / "results" / "qc" / "extraction_failures.csv.gz")
    assert failures.loc[0, "qc_summary"] == str(qc_path)


def test_feature_schema_validation_requires_matched_subject_and_window_cohorts(
    tmp_path,
):
    settings = replace(load_settings(), root=tmp_path)
    manifest = load_manifest(settings.manifest_path)
    definitions = {
        modality: feature_schema(modality)
        for modality in ("eeg", "eog", "emg", "ecg")
    }
    all_names = sorted({
        name for modality_definitions in definitions.values()
        for name in modality_definitions
    })
    subject_rows = []
    window_rows = []
    for participant in manifest["records"]:
        record_id = participant["record_id"]
        row = {
            "record_id": record_id,
            "subject_id": record_id,
            "label": participant["label"],
        }
        for modality, channel in (
            ("eeg", "F3_C3"), ("eog", "ROC_LOC"),
            ("emg", "EMG1_EMG2"), ("ecg", "ECG1_ECG2"),
        ):
            row[f"{modality}__{channel}__mean_median"] = 1.0
            window = {
                "record_id": record_id,
                "subject_id": record_id,
                "label": participant["label"],
                "window_index": 0,
                "window_start_seconds": 0,
                "channel": channel,
                "modality": modality,
                "physical_unit": "uV" if modality != "ecg" else "mV",
                "native_sfreq_hz": 128,
                "qc_is_flat": False,
                "qc_nonfinite_fraction": 0.0,
                "qc_saturated_fraction": 0.0,
                "qc_channel_excluded": False,
            }
            for name in all_names:
                window[f"feature__{name}"] = (
                    1.0 if name in definitions[modality] else float("nan")
                )
            window_rows.append(window)
        subject_rows.append(row)

    feature_dir = tmp_path / "results" / "features"
    feature_dir.mkdir(parents=True)
    pd.DataFrame(subject_rows).to_csv(
        feature_dir / "physiology_features.csv.gz",
        index=False,
        compression="gzip",
    )
    pd.DataFrame(window_rows).to_csv(
        feature_dir / "physiology_window_features.csv.gz",
        index=False,
        compression="gzip",
    )
    documented = {
        f"feature__{name}": {
            "by_modality": {
                modality: definitions[modality][name]
                for modality in definitions
                if name in definitions[modality]
            }
        }
        for name in all_names
    }
    (feature_dir / "physiology_feature_schema.json").write_text(
        json.dumps({
            "channel_features": definitions,
            "window_level": {"feature_columns": documented},
        }),
        encoding="utf-8",
    )

    report = pipeline.validate_feature_schema(settings)

    assert report["status"] == "validated"
    assert report["row_count"] == 88
    assert report["window_row_count"] == 352
