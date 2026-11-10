from __future__ import annotations

import csv
from datetime import datetime, time, timezone
import gzip
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import sys
import tempfile

import numpy as np
import pandas as pd
import requests

from .annotations import cap_features, parse_annotation_text
from .cohort import load_manifest, verify_manifest
from .config import Settings
from .dataset import (
    MODALITY_PREFIXES,
    dataset_digest,
    feature_columns,
    load_physiology_dataset,
    validate_modality_datasets,
)
from .download import download_recording, fetch_edf_header
from .edf import classify_channel, extract_recording, parse_edf_header, read_edf_header
from .features import feature_schema
from .modeling import evaluate_cap_models, evaluate_primary_models


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")
        temporary = Path(stream.name)
    temporary.replace(path)


def _package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "not-installed"


def _owned_temporary(path_value: str | None, directory: Path, record_id: str) -> Path | None:
    if not path_value:
        return None
    path = Path(path_value).resolve()
    expected_prefix = f".capsleep_{record_id}_"
    if path.parent != directory.resolve() or not path.name.startswith(expected_prefix):
        raise ValueError(f"Refusing to use a non-pipeline temporary EDF path: {path}")
    return path if path.is_file() else None


def _atomic_gzip_csv(path: Path, rows: list[dict], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "wb", dir=path.parent, suffix=".csv.gz", delete=False
    ) as stream:
        temporary = Path(stream.name)
    try:
        pd.DataFrame(rows, columns=columns).to_csv(
            temporary, index=False, compression="gzip"
        )
        temporary.replace(path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _combine_window_checkpoints(
    checkpoint_paths: list[Path], destination: Path, columns: list[str]
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "wb", dir=destination.parent, suffix=".csv.gz", delete=False
    ) as stream:
        temporary = Path(stream.name)
    try:
        with gzip.open(temporary, "wt", encoding="utf-8", newline="") as output:
            writer = csv.writer(output)
            writer.writerow(columns)
            for checkpoint_path in checkpoint_paths:
                with gzip.open(checkpoint_path, "rt", encoding="utf-8", newline="") as source:
                    reader = csv.reader(source)
                    header = next(reader, None)
                    if header != columns:
                        raise ValueError(
                            f"Window checkpoint schema changed for {checkpoint_path}"
                        )
                    writer.writerows(reader)
        temporary.replace(destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _write_extraction_failure_log(checkpoint: dict, path: Path) -> None:
    failures = [
        {
            "record_id": record_id,
            "error": record.get("error", ""),
            "qc_summary": record.get("qc_summary", ""),
            "failed_at_utc": record.get("failed_at_utc", ""),
        }
        for record_id, record in checkpoint.get("records", {}).items()
        if record.get("status") == "failed"
    ]
    _atomic_gzip_csv(
        path,
        failures,
        ["record_id", "error", "qc_summary", "failed_at_utc"],
    )


def extract_cohort(settings: Settings) -> dict:
    cohort_report = verify_manifest(settings.manifest_path)
    manifest = load_manifest(settings.manifest_path)
    root = settings.root
    checkpoint_dir = root / "results" / "checkpoints"
    record_dir = checkpoint_dir / "records"
    temporary_dir = root / "data" / "raw" / "_capsleep_tmp"
    checkpoint_path = checkpoint_dir / "extraction_checkpoint.json"
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8")) if checkpoint_path.exists() else {
        "format_version": 2,
        "manifest_sha256": manifest["inventory_sha256"],
        "records": {},
    }
    if checkpoint.get("format_version") != 2:
        raise ValueError(
            "Existing extraction checkpoint uses an older format; preserve it and "
            "choose a separate checkpoint path before restarting"
        )
    if checkpoint.get("manifest_sha256") != manifest["inventory_sha256"]:
        raise ValueError("Checkpoint belongs to a different cohort manifest; inspect before resuming")
    checkpoint.setdefault("records", {})
    checkpoint["cohort_verification"] = cohort_report
    _atomic_json(checkpoint_path, checkpoint)
    for participant in manifest["records"]:
        record_id = participant["record_id"]
        current = checkpoint["records"].get(record_id, {})
        if current.get("status") == "complete":
            saved = record_dir / f"{record_id}.json"
            windows_saved = record_dir / f"{record_id}_windows.csv.gz"
            if saved.exists() and windows_saved.exists():
                continue
            current = {}
        edf_path = _owned_temporary(current.get("temporary_edf"), temporary_dir, record_id)
        failure_qc_path = None
        try:
            if edf_path is None:
                checkpoint["records"][record_id] = {"status": "downloading"}
                _atomic_json(checkpoint_path, checkpoint)
                edf_path = download_recording(
                    f"{settings.base_url}/{participant['filename']}",
                    temporary_dir,
                    settings.minimum_free_gb,
                    settings.timeout_seconds,
                )
                checkpoint["records"][record_id] = {
                    "status": "downloaded",
                    "temporary_edf": str(edf_path),
                }
                _atomic_json(checkpoint_path, checkpoint)
            row, qc, channel_rows, window_rows = extract_recording(
                edf_path, record_id, participant["label"], settings
            )
            if qc["status"] != "complete":
                failure_qc_path = record_dir / f"{record_id}_failure_qc.json"
                _atomic_json(
                    failure_qc_path,
                    {"recording_qc": qc, "channels": channel_rows},
                )
                raise ValueError(qc.get("failure_reason", "Recording QC failed"))
            all_definitions = {
                name
                for modality in ("eeg", "eog", "emg", "ecg")
                for name in feature_schema(modality)
            }
            window_columns = [
                "record_id", "subject_id", "label", "window_index",
                "window_start_seconds", "channel", "modality", "physical_unit",
                "native_sfreq_hz", "qc_is_flat", "qc_nonfinite_fraction",
                "qc_saturated_fraction", "qc_channel_excluded",
                *[f"feature__{name}" for name in sorted(all_definitions)],
            ]
            for window_row in window_rows:
                for name in all_definitions:
                    window_row.setdefault(f"feature__{name}", float("nan"))
            _atomic_gzip_csv(
                record_dir / f"{record_id}_windows.csv.gz",
                window_rows,
                window_columns,
            )
            _atomic_json(
                record_dir / f"{record_id}.json",
                {"physiology": row, "qc": qc, "channels": channel_rows},
            )
            (record_dir / f"{record_id}_failure_qc.json").unlink(missing_ok=True)
            edf_path.unlink()
            checkpoint["records"][record_id] = {
                "status": "complete",
                "window_row_count": len(window_rows),
            }
            _atomic_json(checkpoint_path, checkpoint)
        except Exception as error:
            failure_record = {
                "status": "failed",
                "temporary_edf": str(edf_path) if edf_path and edf_path.exists() else None,
                "error": f"{type(error).__name__}: {error}",
                "failed_at_utc": datetime.now(timezone.utc).isoformat(),
            }
            if failure_qc_path is not None:
                failure_record["qc_summary"] = str(failure_qc_path)
            checkpoint["records"][record_id] = failure_record
            _atomic_json(checkpoint_path, checkpoint)
            _write_extraction_failure_log(
                checkpoint, root / "results" / "qc" / "extraction_failures.csv.gz"
            )
    failed_records = {
        record_id: record.get("error", "unspecified extraction failure")
        for record_id, record in checkpoint["records"].items()
        if record.get("status") != "complete"
    }
    if failed_records:
        failure_summary = "; ".join(
            f"{record_id}: {error}" for record_id, error in failed_records.items()
        )
        raise RuntimeError(
            "Physiology extraction attempted the full cohort but is incomplete; "
            f"no final feature dataset was written. Failures: {failure_summary}. "
            f"Checkpoint: {checkpoint_path}"
        )
    payloads = [json.loads((record_dir / f"{row['record_id']}.json").read_text())
                for row in manifest["records"]]
    if len(payloads) != settings.expected_records:
        raise RuntimeError("Checkpoint data do not contain the full expected cohort")
    expected_ids = [record["record_id"] for record in manifest["records"]]
    completed_ids = [
        item["physiology"]["record_id"] for item in payloads
    ]
    if completed_ids != expected_ids:
        raise RuntimeError("Physiology checkpoints do not match the ordered verified cohort")
    feature_dir = root / "results" / "features"
    qc_dir = root / "results" / "qc"
    feature_dir.mkdir(parents=True, exist_ok=True)
    qc_dir.mkdir(parents=True, exist_ok=True)
    physiology = pd.DataFrame([item["physiology"] for item in payloads])
    physiology_path = feature_dir / "physiology_features.csv.gz"
    physiology.to_csv(physiology_path, index=False, compression="gzip")
    definitions = {
        modality: feature_schema(modality)
        for modality in ("eeg", "eog", "emg", "ecg")
    }
    all_feature_names = sorted({
        name for modality_definitions in definitions.values()
        for name in modality_definitions
    })
    window_columns = [
        "record_id", "subject_id", "label", "window_index",
        "window_start_seconds", "channel", "modality", "physical_unit",
        "native_sfreq_hz", "qc_is_flat", "qc_nonfinite_fraction",
        "qc_saturated_fraction", "qc_channel_excluded",
        *[f"feature__{name}" for name in all_feature_names],
    ]
    window_paths = [
        record_dir / f"{record_id}_windows.csv.gz" for record_id in expected_ids
    ]
    if not all(path.exists() for path in window_paths):
        raise RuntimeError("Window-level checkpoints do not contain all 88 recordings")
    window_path = feature_dir / "physiology_window_features.csv.gz"
    _combine_window_checkpoints(window_paths, window_path, window_columns)
    pd.DataFrame([item["qc"] for item in payloads]).to_csv(
        qc_dir / "recording_qc.csv", index=False
    )
    channel_inventory = []
    for item in payloads:
        for channel in item["channels"]:
            row = dict(channel)
            row.setdefault("feature_channel", row["channel"])
            channel_inventory.append(row)
    pd.DataFrame(channel_inventory).to_csv(
        qc_dir / "channel_inventory.csv", index=False
    )
    _write_extraction_failure_log(
        checkpoint, qc_dir / "extraction_failures.csv.gz"
    )
    _atomic_json(feature_dir / "physiology_feature_schema.json", {
        "signal_normalization": {
            "eeg": "microvolts",
            "eog": "microvolts",
            "emg": "microvolts",
            "ecg": "millivolts",
            "method": "MNE EDF physical-unit conversion followed by SI-to-feature-unit scaling",
        },
        "channel_features": definitions,
        "window_level": {
            "window_seconds": settings.epoch_seconds,
            "feature_columns": {
                f"feature__{name}": {
                    "by_modality": {
                        modality: definitions[modality][name]
                        for modality in definitions
                        if name in definitions[modality]
                    },
                }
                for name in all_feature_names
            },
            "quality_columns": {
                "qc_is_flat": "Boolean: raw window standard deviation is at or below flat_std_threshold.",
                "qc_nonfinite_fraction": "Fraction of non-finite raw samples in this window before interpolation.",
                "qc_saturated_fraction": "Fraction of raw samples at or above the configured modality saturation limit.",
                "qc_channel_excluded": "Boolean: channel exceeded a configured recording-level QC threshold.",
            },
        },
        "subject_level": {
            "feature_name_pattern": "{modality}__{unique_feature_channel}__{feature}_{median|iqr}",
            "aggregation": "Median and interquartile range across non-flat, complete 30-second windows for each channel.",
            "features": all_feature_names,
        },
        "native_sampling_rates": True,
        "source_channel_units_preserved_in": "results/qc/channel_inventory.csv",
    })
    if len(physiology) != settings.expected_records:
        raise RuntimeError("Physiology feature dataset is incomplete")
    _atomic_json(root / "results" / "run_metadata.json", {
        "python": sys.version,
        "packages": {
            name: _package_version(name)
            for name in ("numpy", "scipy", "pandas", "scikit-learn", "mne", "requests")
        },
        "processing": {
            "epoch_seconds": settings.epoch_seconds,
            "chunk_seconds": settings.chunk_seconds,
            "filter_edge_seconds": settings.filter_edge_seconds,
            "filter_order": settings.filter_order,
            "filters_hz": {
                modality: list(band) for modality, band in settings.filters.items()
            },
            "flat_std_threshold": settings.flat_std_threshold,
            "maximum_nonfinite_fraction": settings.maximum_nonfinite_fraction,
            "maximum_flat_window_fraction": settings.maximum_flat_window_fraction,
            "maximum_saturated_fraction": settings.maximum_saturated_fraction,
            "saturation_limits": settings.saturation_limits,
            "native_sampling_rates": True,
            "mne_preload": False,
        },
        "cohort_manifest_sha256": manifest["inventory_sha256"],
        "record_count": len(physiology),
        "window_row_count": sum(
            int(item.get("window_row_count", 0))
            for item in checkpoint["records"].values()
            if item.get("status") == "complete"
        ),
    })
    return {
        "status": "complete",
        "physiology_rows": len(physiology),
        "physiology_path": str(physiology_path),
        "window_feature_path": str(window_path),
        "annotation_validation": "independent; not run by physiological extraction",
    }


def validate_feature_schema(settings: Settings) -> dict:
    path = settings.root / "results" / "features" / "physiology_features.csv.gz"
    window_path = settings.root / "results" / "features" / "physiology_window_features.csv.gz"
    schema_path = settings.root / "results" / "features" / "physiology_feature_schema.json"
    frame = load_physiology_dataset(path, settings.expected_records)
    modalities = validate_modality_datasets(frame)
    manifest = load_manifest(settings.manifest_path)
    expected = {item["record_id"]: item["label"] for item in manifest["records"]}
    observed = dict(zip(frame["record_id"], frame["label"], strict=True))
    if observed != expected:
        raise ValueError("Feature dataset participants or labels differ from the verified manifest")
    if not window_path.exists() or not schema_path.exists():
        raise ValueError("Window-level physiology features and feature definitions are required")
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    window_feature_columns = [
        column for column in pd.read_csv(window_path, nrows=0).columns
        if column.startswith("feature__")
    ]
    defined_window_columns = schema.get("window_level", {}).get(
        "feature_columns", {}
    )
    if set(window_feature_columns) != set(defined_window_columns):
        raise ValueError("Window feature columns do not match the documented feature schema")
    window_ids: set[str] = set()
    eeg_window_ids: set[str] = set()
    window_row_count = 0
    for chunk in pd.read_csv(window_path, chunksize=50000):
        window_row_count += len(chunk)
        feature_values = chunk[window_feature_columns].to_numpy(dtype=float)
        if np.isinf(feature_values).any():
            raise ValueError("Window-level physiology features contain infinite values")
        window_ids.update(chunk["record_id"].astype(str))
        eeg_window_ids.update(
            chunk.loc[chunk["modality"].eq("eeg"), "record_id"].astype(str)
        )
    if window_ids != set(expected) or eeg_window_ids != set(expected):
        raise ValueError("Window features must include EEG windows for the complete verified cohort")
    for column, definition in defined_window_columns.items():
        if not column.startswith("feature__") or not definition.get("by_modality"):
            raise ValueError(f"Undocumented window feature column: {column}")
        if any(
            not details.get("unit") or not details.get("calculation")
            for details in definition["by_modality"].values()
        ):
            raise ValueError(f"Window feature lacks a unit or calculation: {column}")
    for column in feature_columns(frame, MODALITY_PREFIXES):
        matching = [
            name for modality_definitions in schema["channel_features"].values()
            for name in modality_definitions
            if column.endswith(f"__{name}_median") or column.endswith(f"__{name}_iqr")
        ]
        if not matching:
            raise ValueError(f"Subject-level feature is not documented: {column}")
    report = {
        "status": "validated",
        "row_count": len(frame),
        "window_row_count": window_row_count,
        "feature_counts": {name: len(columns) for name, columns in modalities.items()},
        "primary_feature_sources": "physiology only; no annotations",
        "dataset_sha256": dataset_digest(path),
        "window_dataset_sha256": dataset_digest(window_path),
        "feature_schema_sha256": dataset_digest(schema_path),
        "validated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    _atomic_json(settings.root / "results" / "schema_validation.json", report)
    return report


def train_primary(settings: Settings) -> dict:
    feature_path = settings.root / "results" / "features" / "physiology_features.csv.gz"
    window_path = settings.root / "results" / "features" / "physiology_window_features.csv.gz"
    schema_path = settings.root / "results" / "features" / "physiology_feature_schema.json"
    marker_path = settings.root / "results" / "schema_validation.json"
    if not marker_path.exists():
        raise ValueError("Run validate-schema successfully before training primary classifiers")
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    if marker.get("dataset_sha256") != dataset_digest(feature_path):
        raise ValueError("Physiology dataset changed after schema validation; validate it again")
    if (
        marker.get("window_dataset_sha256") != dataset_digest(window_path)
        or marker.get("feature_schema_sha256") != dataset_digest(schema_path)
    ):
        raise ValueError("Physiology windows or feature schema changed after validation")
    frame = load_physiology_dataset(feature_path, settings.expected_records)
    return evaluate_primary_models(
        frame,
        settings.root / "results" / "models" / "primary",
        settings.folds,
        settings.random_seed,
        settings.trees,
    )


def analyze_cap(settings: Settings) -> dict:
    path = settings.root / "results" / "features" / "cap_features.csv.gz"
    validation_path = settings.root / "results" / "cap_validation.json"
    if not validation_path.exists():
        raise ValueError("Run validate-annotations or a complete extraction before CAP analysis")
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    if (
        validation.get("status") != "validated"
        or validation.get("dataset_sha256") != dataset_digest(path)
    ):
        raise ValueError("CAP feature dataset is incomplete or changed after annotation validation")
    frame = pd.read_csv(path)
    if len(frame) != settings.expected_records or frame["record_id"].nunique() != settings.expected_records:
        raise ValueError(
            "CAP analysis requires validated annotations for all 88 recordings; "
            "review results/qc/annotation_qc.csv"
        )
    manifest = load_manifest(settings.manifest_path)
    expected = {item["record_id"]: item["label"] for item in manifest["records"]}
    if dict(zip(frame["record_id"], frame["label"], strict=True)) != expected:
        raise ValueError("CAP feature dataset participants or labels differ from the verified manifest")
    numeric = frame.select_dtypes(include="number").columns.tolist()
    summary = frame.groupby("label", observed=True)[numeric].agg(["mean", "std", "median"])
    summary.columns = [f"{feature}_{statistic}" for feature, statistic in summary.columns]
    summary.to_csv(settings.root / "results" / "cap_group_summary.csv")
    return {"status": "complete", "participants": len(frame), "features": numeric}


def train_cap(settings: Settings) -> dict:
    physiology_path = settings.root / "results" / "features" / "physiology_features.csv.gz"
    window_path = settings.root / "results" / "features" / "physiology_window_features.csv.gz"
    feature_schema_path = settings.root / "results" / "features" / "physiology_feature_schema.json"
    cap_path = settings.root / "results" / "features" / "cap_features.csv.gz"
    marker_path = settings.root / "results" / "schema_validation.json"
    cap_marker_path = settings.root / "results" / "cap_validation.json"
    if not cap_marker_path.exists():
        raise ValueError("Run validate-annotations or a complete extraction before CAP training")
    cap_marker = json.loads(cap_marker_path.read_text(encoding="utf-8"))
    if (
        cap_marker.get("status") != "validated"
        or cap_marker.get("dataset_sha256") != dataset_digest(cap_path)
    ):
        raise ValueError("CAP dataset is incomplete or changed after annotation validation")
    if not marker_path.exists():
        raise ValueError("Run validate-schema successfully before training CAP-informed classifiers")
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    if marker.get("dataset_sha256") != dataset_digest(physiology_path):
        raise ValueError("Physiology dataset changed after schema validation; validate it again")
    if (
        marker.get("window_dataset_sha256") != dataset_digest(window_path)
        or marker.get("feature_schema_sha256") != dataset_digest(feature_schema_path)
    ):
        raise ValueError("Physiology windows or feature schema changed after validation")
    physiology = load_physiology_dataset(physiology_path, settings.expected_records)
    cap = pd.read_csv(cap_path)
    if len(cap) != settings.expected_records:
        raise ValueError("CAP-informed experiment requires complete 88-participant annotations")
    cap_columns = [
        column for column in cap.columns
        if column not in {"record_id", "subject_id", "label"}
        and pd.api.types.is_numeric_dtype(cap[column])
    ]
    return evaluate_cap_models(
        physiology, cap, cap_columns, settings.root / "results" / "models" / "cap",
        settings.folds, settings.random_seed, settings.trees,
    )


def validate_annotation_cohort(settings: Settings) -> dict:
    verify_manifest(settings.manifest_path)
    manifest = load_manifest(settings.manifest_path)
    metadata_path = settings.root / "results" / "qc" / "recording_metadata.csv"
    if not metadata_path.exists():
        raise ValueError("Run record-inventory before validating CAP annotations")
    metadata = pd.read_csv(metadata_path)
    expected_ids = [record["record_id"] for record in manifest["records"]]
    observed_ids = metadata["record_id"].tolist()
    if observed_ids != expected_ids:
        raise ValueError("Header metadata does not match the ordered verified cohort manifest")
    validation_path = settings.root / "results" / "cap_validation.json"
    _atomic_json(validation_path, {"status": "in_progress", "row_count": 0})
    cap_rows = []
    qc_rows = []
    for participant, recording in zip(manifest["records"], metadata.to_dict("records"), strict=True):
        record_id = participant["record_id"]
        events = []
        try:
            response = requests.get(
                f"{settings.base_url}/{record_id}.txt", timeout=settings.timeout_seconds
            )
            response.raise_for_status()
            text = response.content.decode("utf-8-sig")
            events = parse_annotation_text(
                text, recording["edf_start_time"], recording["duration_seconds"]
            )
            features = cap_features(events, recording["duration_seconds"])
            cap_rows.append({
                "record_id": record_id,
                "subject_id": record_id,
                "label": participant["label"],
                **features,
            })
            qc_rows.append({
                "record_id": record_id,
                "status": "complete",
                "event_count": len(events),
                "annotation_url": f"{settings.base_url}/{record_id}.txt",
                "error": "",
                **features,
            })
        except (requests.RequestException, UnicodeDecodeError, ValueError) as error:
            qc_rows.append({
                "record_id": record_id,
                "status": "failed",
                "event_count": len(events),
                "annotation_url": f"{settings.base_url}/{record_id}.txt",
                "error": f"{type(error).__name__}: {error}",
            })
    qc = pd.DataFrame(qc_rows)
    qc_path = settings.root / "results" / "qc" / "annotation_qc.csv"
    qc_path.parent.mkdir(parents=True, exist_ok=True)
    qc.to_csv(qc_path, index=False)
    if len(cap_rows) != settings.expected_records or (qc["status"] != "complete").any():
        report = {
            "status": "incomplete",
            "row_count": len(cap_rows),
            "failed_annotations": int((qc["status"] != "complete").sum()),
            "qc_path": str(qc_path),
        }
        _atomic_json(validation_path, report)
        raise RuntimeError(
            f"CAP annotation validation failed for "
            f"{report['failed_annotations']} of {settings.expected_records} recordings; "
            f"review {qc_path}"
        )
    cap = pd.DataFrame(cap_rows)
    output = settings.root / "results" / "features" / "cap_features.csv.gz"
    output.parent.mkdir(parents=True, exist_ok=True)
    cap.to_csv(output, index=False, compression="gzip")
    report = {
        "status": "validated",
        "row_count": len(cap),
        "failed_annotations": 0,
        "dataset_sha256": dataset_digest(output),
        "validated_at_utc": datetime.now(timezone.utc).isoformat(),
        "qc_path": str(qc_path),
    }
    _atomic_json(validation_path, report)
    return report


def diagnose_annotation_cohort(settings: Settings) -> dict:
    qc_path = settings.root / "results" / "qc" / "annotation_qc.csv"
    metadata_path = settings.root / "results" / "qc" / "recording_metadata.csv"
    validation_path = settings.root / "results" / "cap_validation.json"
    if not qc_path.exists() or not metadata_path.exists() or not validation_path.exists():
        raise ValueError("Run record-inventory and validate-annotations before diagnosis")
    qc = pd.read_csv(qc_path).fillna("")
    metadata = pd.read_csv(metadata_path).set_index("record_id")
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    failed = qc.loc[qc["status"].eq("failed")]
    if len(qc) != settings.expected_records:
        raise ValueError("Annotation QC does not contain the full expected cohort")
    evidence_rows = []
    for item in failed.to_dict("records"):
        record_id = item["record_id"]
        response = requests.get(
            f"{settings.base_url}/{record_id}.txt", timeout=settings.timeout_seconds
        )
        response.raise_for_status()
        text = response.content.decode("utf-8-sig")
        lines = text.splitlines()
        header_line_number = next(
            (index for index, line in enumerate(lines) if line.startswith("Sleep Stage\t")),
            None,
        )
        if header_line_number is None:
            raise ValueError(f"{record_id}: source annotation has no recognizable header")
        header = [name.strip().lower() for name in lines[header_line_number].split("\t")]
        column = {
            "stage": next(i for i, name in enumerate(header) if name.startswith("sleep stage")),
            "clock": next(i for i, name in enumerate(header) if name.startswith("time")),
            "event": next(i for i, name in enumerate(header) if name.startswith("event")),
            "duration": next(i for i, name in enumerate(header) if name.startswith("duration")),
        }
        source_events = []
        malformed = []
        for line_number, line in enumerate(lines[header_line_number + 1:], header_line_number + 2):
            if not line.strip():
                continue
            fields = line.split("\t")
            if len(fields) != len(header):
                malformed.append((line_number, line))
                continue
            try:
                clock_text = fields[column["clock"]]
                parsed_clock = time.fromisoformat(clock_text.replace(".", ":"))
                clock_seconds = (
                    parsed_clock.hour * 3600 + parsed_clock.minute * 60 + parsed_clock.second
                )
                duration = float(fields[column["duration"]])
            except ValueError:
                malformed.append((line_number, line))
                continue
            source_events.append({
                "line": line_number,
                "raw": line,
                "clock": clock_seconds,
                "clock_text": clock_text,
                "stage": fields[column["stage"]].strip(),
                "event": fields[column["event"]].strip(),
                "duration": duration,
            })
        if source_events:
            first_clock = source_events[0]["clock"]
            day_offset = 0
            previous_clock = first_clock
            for event in source_events:
                if previous_clock - event["clock"] > 43200:
                    day_offset += 86400
                event["relative_start"] = event["clock"] + day_offset - first_clock
                previous_clock = event["clock"]
            annotation_start = time.fromisoformat(
                str(metadata.loc[record_id, "edf_start_time"]).replace(".", ":")
            )
            edf_start_seconds = (
                annotation_start.hour * 3600
                + annotation_start.minute * 60
                + annotation_start.second
            )
            offset_seconds = (first_clock - edf_start_seconds + 43200) % 86400 - 43200
            stage_events = [
                event for event in source_events
                if event["event"].startswith("SLEEP-")
            ]
            stage_seconds = sum(event["duration"] for event in stage_events)
            stage_coverage = (
                100 * stage_seconds / float(metadata.loc[record_id, "duration_seconds"])
            )
            span_seconds = max(
                event["relative_start"] + event["duration"]
                for event in source_events
            )
        else:
            offset_seconds = float("nan")
            stage_events = []
            stage_coverage = 0.0
            span_seconds = 0.0
        cap_events = sorted(
            (
                event for event in source_events
                if event["event"].startswith("MCAP-A")
            ),
            key=lambda event: event["relative_start"],
        )
        cap_duration_issues = [
            event for event in cap_events
            if not 2.0 <= event["duration"] <= 60.0
        ]
        overlaps = [
            (previous, current)
            for previous, current in zip(cap_events, cap_events[1:])
            if current["relative_start"] < previous["relative_start"] + previous["duration"]
        ]
        error = str(item["error"])
        if "Sleep-stage annotation covers" in error:
            category = "source scoring coverage below 80%"
            details = (
                f"{stage_coverage:.1f}% ({len(stage_events)} scored stage rows; "
                f"{stage_seconds:g}s of {float(metadata.loc[record_id, 'duration_seconds']):g}s EDF)"
            )
        elif "CAP-A duration" in error:
            category = "source CAP-A duration outside configured range"
            event = cap_duration_issues[0] if cap_duration_issues else None
            details = f"line {event['line']}: {event['raw']}" if event else error
        elif "overlap" in error:
            category = "overlapping source CAP events"
            details = (
                f"lines {overlaps[0][0]['line']} and {overlaps[0][1]['line']}: "
                f"{overlaps[0][0]['raw']} / {overlaps[0][1]['raw']}"
                if overlaps else error
            )
        elif "Malformed annotation line" in error:
            category = "malformed/truncated source annotation row"
            details = (
                f"line {malformed[0][0]}: {malformed[0][1]}"
                if malformed else error
            )
        elif "Annotation time span" in error:
            category = "annotation span exceeds EDF duration"
            stage_lines = [
                event for event in source_events if event["event"].startswith("SLEEP-")
            ]
            details = (
                f"annotation span {span_seconds:g}s vs EDF "
                f"{float(metadata.loc[record_id, 'duration_seconds']):g}s; "
                f"last scored row line {stage_lines[-1]['line']}: {stage_lines[-1]['raw']}"
                if stage_lines else error
            )
        else:
            category = "other validation failure"
            details = error
        first_stage = stage_events[0] if stage_events else None
        last_stage = stage_events[-1] if stage_events else None
        evidence_rows.append({
            "record_id": record_id,
            "category": category,
            "edf_duration_seconds": metadata.loc[record_id, "duration_seconds"],
            "edf_start_time": metadata.loc[record_id, "edf_start_time"],
            "annotation_first_event_offset_seconds": offset_seconds,
            "annotation_span_seconds": span_seconds,
            "stage_coverage_percent": stage_coverage,
            "first_stage_row": f"line {first_stage['line']}: {first_stage['raw']}" if first_stage else "",
            "last_stage_row": f"line {last_stage['line']}: {last_stage['raw']}" if last_stage else "",
            "details": details,
        })
    evidence = pd.DataFrame(evidence_rows)
    if len(evidence) != len(failed):
        raise RuntimeError("Could not produce source-level evidence for every failed annotation")
    report_path = settings.root / "results" / "qc" / "annotation_diagnostic.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    counts = evidence["category"].value_counts().to_dict()
    lines = [
        "# CAP annotation failure diagnostic",
        "",
        f"- Validation status: `{validation.get('status')}`",
        f"- Passed: {int(qc['status'].eq('complete').sum())}/{len(qc)}",
        f"- Blocked: {len(failed)}/{len(qc)}",
        "- Source annotation text was read-only; no annotation was altered, repaired, clipped, or imputed.",
        "- Diagnosis is independent of physiological feature extraction.",
        "",
        "## Failure groups",
        "",
        "| Category | Count |",
        "|---|---:|",
    ]
    lines.extend(f"| {category} | {count} |" for category, count in counts.items())
    lines.extend([
        "",
        "## Per-record evidence",
        "",
        "| Record | Failure | EDF duration (s) | EDF start | Annotation first-event offset (s) | Stage coverage (%) | Source evidence |",
        "|---|---|---:|---|---:|---:|---|",
    ])
    for row in evidence.to_dict("records"):
        evidence_text = str(row["details"]).replace("|", "\\|").replace("\n", " ")
        lines.append(
            f"| {row['record_id']} | {row['category']} | "
            f"{row['edf_duration_seconds']} | {row['edf_start_time']} | "
            f"{row['annotation_first_event_offset_seconds']:.0f} | "
            f"{row['stage_coverage_percent']:.1f} | `{evidence_text}` |"
        )
    lines.extend([
        "",
        "## Assessment and proposed resolution",
        "",
        "- **Parser/schema:** the earlier five-column versus six-column export mismatch is fixed and is not among the 16 remaining failures. The `nfle27` row is literally truncated in the source text; strict rejection is appropriate. Re-fetch the source or request a corrected export from the database curator, then rerun validation. Do not synthesize missing timestamp, event, or duration fields.",
        "- **Scoring coverage:** eleven records have less than the existing 80% explicit 30-second stage coverage threshold. This is source coverage, not an EDF parser failure. Review the annotation and EDF provenance with the database curator; do not extend stage labels or lower the threshold to force inclusion.",
        "- **CAP event duration/timing:** `n10` has a source CAP-A duration of 65 seconds, and `n14` has overlapping CAP-A source rows. Confirm whether these reflect valid scorer export conventions or source defects against authoritative scoring. Preserve the rows and keep these records blocked until adjudicated; do not truncate durations or reorder/merge overlapping events.",
        "- **Annotation/EDF span:** `plm1` and `rbd9` have scored annotation spans longer than the corresponding EDF durations. Verify the source EDF duration and annotation/recording pairing. Do not clip terminal events or alter the CAP denominator without source confirmation.",
        "- **Physiological analysis:** all these annotation defects are confined to the secondary CAP workflow. They must not exclude participants from EEG/EMG/EOG/ECG feature extraction. CAP descriptive analysis and CAP-informed classification remain gated until all 88 source records pass validation.",
        "",
        "Source files: PhysioNet CAP Sleep Database `1.0.0` annotation exports; EDF start times and durations are from `recording_metadata.csv`.",
    ])
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    evidence.to_csv(
        settings.root / "results" / "qc" / "annotation_diagnostic.csv",
        index=False,
    )
    return {
        "status": "diagnosed",
        "passed": int(qc["status"].eq("complete").sum()),
        "blocked": len(failed),
        "failure_groups": counts,
        "report_path": str(report_path),
    }


def record_inventory(settings: Settings) -> dict:
    verify_manifest(settings.manifest_path)
    manifest = load_manifest(settings.manifest_path)
    inventory = []
    channel_rows = []
    for record in manifest["records"]:
        url = f"{settings.base_url}/{record['filename']}"
        header, file_size = fetch_edf_header(url, settings.timeout_seconds)
        duration, record_duration, channels = parse_edf_header(header, file_size)
        modalities = {
            modality for name, _, _ in channels
            if (modality := classify_channel(name)) is not None
        }
        missing = [name for name in ("eeg", "eog", "emg", "ecg") if name not in modalities]
        inventory.append({
            **record,
            "remote_size_bytes": file_size,
            "duration_seconds": duration,
            "edf_record_duration_seconds": record_duration,
            "edf_channel_count": len(channels),
            "recognized_modalities": "|".join(sorted(modalities)),
            "missing_modalities": "|".join(missing),
            "edf_start_date": header[168:176].decode("ascii").strip(),
            "edf_start_time": header[176:184].decode("ascii").strip(),
            "url": url,
        })
        channel_rows.extend({
            "record_id": record["record_id"],
            "label": record["label"],
            "channel": name,
            "physical_unit": unit,
            "native_sfreq_hz": sfreq,
            "modality": classify_channel(name) or "",
        } for name, unit, sfreq in channels)
    if len(inventory) != settings.expected_records or len(channel_rows) == 0:
        raise RuntimeError("EDF header inventory is incomplete")
    missing_eeg = [
        item["record_id"] for item in inventory
        if "eeg" not in item["recognized_modalities"].split("|")
    ]
    if missing_eeg:
        raise RuntimeError(f"EDF headers lack recognized EEG channels: {missing_eeg}")
    output = settings.root / "results" / "qc" / "recording_inventory.json"
    _atomic_json(output, {"records": inventory, "record_count": len(inventory)})
    pd.DataFrame(inventory).to_csv(output.with_name("recording_metadata.csv"), index=False)
    pd.DataFrame(channel_rows).to_csv(output.with_name("channel_inventory.csv"), index=False)
    return {
        "status": "complete",
        "record_count": len(inventory),
        "channel_count": len(channel_rows),
        "path": str(output),
    }
