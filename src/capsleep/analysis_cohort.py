from __future__ import annotations

import csv
import gzip
import json
import math
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from .config import Settings
from .dataset import dataset_digest, load_physiology_dataset
from .features import feature_schema
from .modeling import evaluate_primary_models


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _json_safe(value):
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _feature_prefix(modality: str, channel: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9]+", "_", channel).strip("_")
    return f"{modality}__{normalized}__"


def _modality_availability(
    channels: list[dict], required_modalities: set[str]
) -> tuple[list[str], list[str]]:
    modality_channels: dict[str, list[dict]] = defaultdict(list)
    for channel in channels:
        modality = channel.get("modality")
        if modality in required_modalities:
            modality_channels[modality].append(channel)
    missing = sorted(required_modalities - set(modality_channels))
    unusable = sorted(
        modality for modality in required_modalities
        if not any(
            not channel.get("excluded", False)
            and int(channel.get("usable_window_count", 0)) > 0
            and str(channel.get("physical_unit", "")).strip()
            for channel in modality_channels.get(modality, [])
        )
    )
    return missing, unusable


def validate_analysis_cohort(
    settings: Settings,
    cohort_config: Path | None = None,
) -> dict:
    config_path = cohort_config or (
        settings.root / "config" / "analysis_cohorts" / "primary_physiology_85_v1.json"
    )
    config = _read_json(config_path)
    analysis_id = config["analysis_id"]
    if not re.fullmatch(r"[A-Za-z0-9_-]+", analysis_id):
        raise ValueError(f"Unsafe analysis ID in cohort configuration: {analysis_id}")
    if not analysis_id.startswith("primary_physiology_"):
        raise ValueError("Analysis ID must start with 'primary_physiology_'")
    output_stem = analysis_id.removeprefix("primary_physiology_")
    manifest_path = settings.root / config["source_manifest"]
    manifest = _read_json(manifest_path)
    source_by_id = {row["record_id"]: row for row in manifest["records"]}
    included = config["included_record_ids"]
    excluded = {row["record_id"]: row for row in config["excluded_records"]}
    included_set = set(included)
    expected_source_ids = set(source_by_id)
    if len(included) != len(included_set):
        raise ValueError("Analysis cohort contains duplicate included record IDs")
    if included_set & set(excluded):
        raise ValueError("Analysis cohort records cannot be both included and excluded")
    if included_set | set(excluded) != expected_source_ids:
        raise ValueError("Analysis cohort must partition the unchanged source manifest")
    if manifest["inventory_sha256"] != config["source_inventory_sha256"]:
        raise ValueError("Analysis cohort references a different source inventory")
    expected_counts = config["expected_class_counts"]
    observed_counts = Counter(source_by_id[record_id]["label"] for record_id in included)
    if dict(observed_counts) != expected_counts:
        raise ValueError(
            f"Analysis cohort class counts differ from configuration: {dict(observed_counts)}"
        )

    checkpoint_path = settings.root / "results" / "checkpoints" / "extraction_checkpoint.json"
    checkpoint = _read_json(checkpoint_path)
    if checkpoint.get("manifest_sha256") != manifest["inventory_sha256"]:
        raise ValueError("Extraction checkpoints do not match the source inventory")

    record_results = []
    feature_rows = []
    record_errors = []
    total_window_rows = 0
    expected_observations = 0
    modality_window_rows: Counter[str] = Counter()
    required_modalities = set(config["eligibility"]["required_modalities"])
    epoch_seconds = int(config["eligibility"]["window_seconds"])
    recording_metadata_path = settings.root / "results" / "qc" / "recording_metadata.csv"
    recording_metadata = (
        pd.read_csv(recording_metadata_path).set_index("record_id")
        if recording_metadata_path.exists()
        else None
    )

    for participant in (source_by_id[record_id] for record_id in included):
        record_id = participant["record_id"]
        record_errors_for_subject = []
        checkpoint_record = checkpoint.get("records", {}).get(record_id, {})
        record_dir = settings.root / "results" / "checkpoints" / "records"
        payload_path = record_dir / f"{record_id}.json"
        windows_path = record_dir / f"{record_id}_windows.csv.gz"
        if (
            checkpoint_record.get("status") != "complete"
            or not payload_path.is_file()
            or not windows_path.is_file()
        ):
            record_errors_for_subject.append("missing_or_incomplete_extraction_checkpoint")
            record_results.append({
                "record_id": record_id,
                "label": participant["label"],
                "status": "blocked",
                "errors": "|".join(record_errors_for_subject),
            })
            record_errors.extend(
                {"record_id": record_id, "reason": reason}
                for reason in record_errors_for_subject
            )
            continue

        payload = _read_json(payload_path)
        physiology = payload["physiology"]
        qc = payload["qc"]
        channels = payload["channels"]
        if physiology.get("record_id") != record_id or physiology.get("label") != participant["label"]:
            record_errors_for_subject.append("subject_checkpoint_identity_or_label_mismatch")
        if qc.get("status") != "complete":
            record_errors_for_subject.append("recording_qc_not_complete")
        if int(qc.get("excluded_channel_count", 0)) > 0:
            record_errors_for_subject.append("one_or_more_channels_failed_qc")
        if qc.get("qc_excluded_channels"):
            record_errors_for_subject.append("qc_excluded_channel_list_not_empty")

        missing_modalities, unusable_modalities = _modality_availability(
            channels, required_modalities
        )
        modality_channels: dict[str, list[dict]] = defaultdict(list)
        for channel in channels:
            if channel.get("modality") in required_modalities:
                modality_channels[channel["modality"]].append(channel)
        if missing_modalities:
            record_errors_for_subject.append(
                "missing_modalities:" + ",".join(missing_modalities)
            )
        if unusable_modalities:
            record_errors_for_subject.append(
                "no_qc_usable_channels:" + ",".join(unusable_modalities)
            )

        duration = float(qc["duration_seconds"])
        if not math.isfinite(duration) or duration <= 0:
            record_errors_for_subject.append("invalid_recording_duration")
            expected_windows_per_channel = 0
        else:
            expected_windows_per_channel = int(duration // epoch_seconds)
        if recording_metadata is not None and record_id in recording_metadata.index:
            metadata_duration = float(
                recording_metadata.loc[record_id, "duration_seconds"]
            )
            if not math.isclose(duration, metadata_duration, rel_tol=0, abs_tol=1e-6):
                record_errors_for_subject.append("duration_mismatch_with_edf_inventory")
        elif recording_metadata is not None:
            record_errors_for_subject.append("recording_missing_from_edf_inventory")

        expected_channel_keys = {
            (
                channel["modality"],
                channel.get("feature_channel", channel["channel"]),
            )
            for modality in required_modalities
            for channel in modality_channels.get(modality, [])
        }
        expected_record_observations = (
            expected_windows_per_channel * len(expected_channel_keys)
        )
        expected_observations += expected_record_observations
        observed_groups: dict[tuple[str, str], list[int]] = defaultdict(list)
        observed_rows = 0
        feature_data: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(
            lambda: defaultdict(list)
        )
        feature_names_by_modality = {
            modality: tuple(feature_schema(modality))
            for modality in required_modalities
        }
        try:
            with gzip.open(
                windows_path, "rt", encoding="utf-8", newline=""
            ) as stream:
                reader = csv.DictReader(stream)
                required_columns = {
                    "record_id", "subject_id", "label", "window_index",
                    "channel", "modality", "qc_is_flat", "qc_channel_excluded",
                }
                if not required_columns.issubset(reader.fieldnames or []):
                    raise ValueError("window_checkpoint_missing_required_columns")
                for row in reader:
                    observed_rows += 1
                    if (
                        row["record_id"] != record_id
                        or row["subject_id"] != record_id
                        or row["label"] != participant["label"]
                    ):
                        record_errors_for_subject.append(
                            "window_observation_identity_or_label_mismatch"
                        )
                        break
                    modality = row["modality"]
                    if modality not in required_modalities:
                        continue
                    key = (modality, row["channel"])
                    index = int(row["window_index"])
                    observed_groups[key].append(index)
                    modality_window_rows[modality] += 1
                    flat = row["qc_is_flat"].strip().lower() in {"true", "1"}
                    excluded_channel = (
                        row["qc_channel_excluded"].strip().lower() in {"true", "1"}
                    )
                    if flat or excluded_channel:
                        continue
                    for feature_name in feature_names_by_modality[modality]:
                        value = row.get(f"feature__{feature_name}", "")
                        feature_data[key][feature_name].append(
                            float(value) if value.strip() else float("nan")
                        )
        except (OSError, ValueError, csv.Error) as error:
            record_errors_for_subject.append(
                f"window_checkpoint_read_error:{type(error).__name__}:{error}"
            )

        total_window_rows += observed_rows
        if observed_rows != int(checkpoint_record.get("window_row_count", -1)):
            record_errors_for_subject.append("window_row_count_differs_from_checkpoint")
        if observed_rows != expected_record_observations:
            record_errors_for_subject.append(
                "window_row_count_differs_from_expected_channel_window_observations"
            )
        if set(observed_groups) != expected_channel_keys:
            record_errors_for_subject.append("window_channel_set_differs_from_channel_inventory")
        for key in expected_channel_keys:
            indices = observed_groups.get(key, [])
            if (
                len(indices) != expected_windows_per_channel
                or len(set(indices)) != expected_windows_per_channel
                or (indices and (min(indices) != 0 or max(indices) != expected_windows_per_channel - 1))
            ):
                record_errors_for_subject.append(
                    f"window_index_or_count_mismatch:{key[0]}:{key[1]}"
                )

        for (modality, channel), feature_values in feature_data.items():
            prefix = _feature_prefix(modality, channel)
            for feature_name, values in feature_values.items():
                if not values:
                    continue
                array = np.asarray(values, dtype=float)
                aggregate_values = {
                    "median": float(np.median(array)),
                    "iqr": float(np.percentile(array, 75) - np.percentile(array, 25)),
                }
                for statistic, value in aggregate_values.items():
                    feature_column = f"{prefix}{feature_name}_{statistic}"
                    saved_value = physiology.get(feature_column)
                    same = (
                        saved_value is not None
                        and (
                            (math.isnan(value) and math.isnan(float(saved_value)))
                            or math.isclose(
                                value, float(saved_value), rel_tol=1e-8, abs_tol=1e-10
                            )
                        )
                    )
                    if not same:
                        record_errors_for_subject.append(
                            f"subject_aggregation_mismatch:{feature_column}"
                        )

        available_subject_modalities = {
            modality for modality in required_modalities
            if any(
                column.startswith(f"{modality}__")
                for column in physiology
            )
        }
        if available_subject_modalities != required_modalities:
            record_errors_for_subject.append("subject_feature_vector_missing_modalities")

        status = "eligible" if not record_errors_for_subject else "blocked"
        record_results.append({
            "record_id": record_id,
            "label": participant["label"],
            "status": status,
            "duration_seconds": duration,
            "expected_windows_per_channel": expected_windows_per_channel,
            "physiological_channel_count": len(expected_channel_keys),
            "expected_channel_window_observations": expected_record_observations,
            "observed_window_rows": observed_rows,
            "missing_modalities": "|".join(missing_modalities),
            "unusable_modalities": "|".join(unusable_modalities),
            "errors": "|".join(dict.fromkeys(record_errors_for_subject)),
        })
        record_errors.extend(
            {"record_id": record_id, "reason": reason}
            for reason in dict.fromkeys(record_errors_for_subject)
        )
        feature_rows.append(physiology)

    candidate_features = pd.DataFrame(feature_rows)
    if not candidate_features.empty:
        candidate_features = candidate_features.reindex(
            columns=[
                "record_id", "subject_id", "label",
                *[
                    column for column in candidate_features.columns
                    if column not in {"record_id", "subject_id", "label"}
                ],
            ]
        )
        candidate_path = (
            settings.root / "results" / "features"
            / f"physiology_primary_{output_stem}_candidate.csv.gz"
        )
        candidate_path.parent.mkdir(parents=True, exist_ok=True)
        candidate_features.to_csv(candidate_path, index=False, compression="gzip")
    else:
        candidate_path = None

    feature_columns = [
        column for column in candidate_features.columns
        if column.startswith(("eeg__", "eog__", "emg__", "ecg__"))
    ] if not candidate_features.empty else []
    missingness = {}
    for modality in ("eeg", "emg", "eog", "ecg"):
        columns = [column for column in feature_columns if column.startswith(f"{modality}__")]
        missingness[modality] = {
            "feature_count": len(columns),
            "all_missing_feature_count": sum(
                candidate_features[column].isna().all() for column in columns
            ),
            "missing_cells": int(candidate_features[columns].isna().sum().sum())
            if columns else 0,
            "total_cells": int(candidate_features[columns].size) if columns else 0,
        }

    fold_preview = []
    if len(candidate_features) == len(included):
        splitter = StratifiedGroupKFold(
            n_splits=int(config["primary_model"]["folds"]),
            shuffle=True,
            random_state=int(config["primary_model"]["random_seed"]),
        )
        try:
            splits = list(splitter.split(
                candidate_features,
                candidate_features["label"],
                groups=candidate_features["subject_id"],
            ))
            for fold_index, (train, test) in enumerate(splits):
                if set(candidate_features.iloc[train]["subject_id"]) & set(
                    candidate_features.iloc[test]["subject_id"]
                ):
                    raise ValueError("subject_leakage")
                fold_preview.append({
                    "fold": fold_index,
                    "train_class_counts": candidate_features.iloc[train]["label"].value_counts().to_dict(),
                    "test_class_counts": candidate_features.iloc[test]["label"].value_counts().to_dict(),
                })
        except ValueError as error:
            record_errors.append({
                "record_id": "",
                "reason": f"cross_validation_split_failure:{error}",
            })

    blocked_records = [
        row["record_id"] for row in record_results if row.get("status") != "eligible"
    ]
    durations = [
        float(row["duration_seconds"])
        for row in record_results if row.get("duration_seconds") is not None
    ]
    report = {
        "analysis_id": analysis_id,
        "analysis_cohort_config_sha256": dataset_digest(config_path),
        "status": "eligible" if not blocked_records and not record_errors else "blocked",
        "classification_started": False,
        "source_manifest_unchanged": True,
        "source_manifest_record_count": len(source_by_id),
        "included_record_count": len(included),
        "excluded_record_count": len(excluded),
        "included_class_counts": dict(observed_counts),
        "additional_blocked_records": blocked_records,
        "error_count": len(record_errors),
        "errors": record_errors,
        "subject_feature_rows": len(candidate_features),
        "unique_subject_count": (
            int(candidate_features["subject_id"].nunique())
            if not candidate_features.empty else 0
        ),
        "subject_feature_columns": len(feature_columns),
        "recording_duration_seconds": {
            "minimum": min(durations) if durations else None,
            "maximum": max(durations) if durations else None,
            "unique_count": len(set(durations)),
        },
        "feature_counts_by_modality": {
            modality: sum(
                column.startswith(f"{modality}__") for column in feature_columns
            )
            for modality in ("eeg", "emg", "eog", "ecg")
        },
        "feature_missingness_by_modality": missingness,
        "window_observation_unit": "one row per recording/channel/30-second window",
        "window_rows_are_independent_participants": False,
        "window_rows_observed": total_window_rows,
        "expected_channel_window_observations": expected_observations,
        "window_rows_equal_expected_observations": (
            total_window_rows == expected_observations
        ),
        "window_row_counts_by_modality": dict(modality_window_rows),
        "subject_aggregation": (
            "Per channel, median and IQR (75th minus 25th percentile) across "
            "non-flat, non-QC-excluded complete 30-second windows."
        ),
        "cross_validation_plan_preview": fold_preview,
        "same_fold_plan_for_all_models": True,
        "candidate_feature_dataset": (
            str(candidate_path.relative_to(settings.root)) if candidate_path else None
        ),
        "candidate_feature_dataset_sha256": (
            dataset_digest(candidate_path) if candidate_path else None
        ),
        "validated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    qc_dir = settings.root / "results" / "qc"
    qc_dir.mkdir(parents=True, exist_ok=True)
    (qc_dir / f"{analysis_id}_preflight.json").write_text(
        json.dumps(_json_safe(report), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    pd.DataFrame(record_results).to_csv(
        qc_dir / f"{analysis_id}_record_validation.csv", index=False
    )
    return _json_safe(report)


def train_analysis_cohort(
    settings: Settings,
    cohort_config: Path | None = None,
) -> dict:
    config_path = cohort_config or (
        settings.root / "config" / "analysis_cohorts" / "primary_physiology_85_v1.json"
    )
    config = _read_json(config_path)
    analysis_id = config["analysis_id"]
    report_path = (
        settings.root / "results" / "qc" / f"{analysis_id}_preflight.json"
    )
    if not report_path.is_file():
        raise ValueError("Run validate-analysis-cohort before training")
    preflight = _read_json(report_path)
    if preflight.get("analysis_id") != analysis_id or preflight.get("status") != "eligible":
        raise ValueError(
            f"Analysis cohort {analysis_id} is not eligible; "
            f"preflight status is {preflight.get('status')!r}"
        )
    if preflight.get("analysis_cohort_config_sha256") != dataset_digest(config_path):
        raise ValueError("Analysis cohort config changed after eligibility validation")
    candidate_path = settings.root / preflight["candidate_feature_dataset"]
    if (
        not candidate_path.is_file()
        or dataset_digest(candidate_path)
        != preflight.get("candidate_feature_dataset_sha256")
    ):
        raise ValueError("Candidate feature dataset changed after cohort validation")
    frame = load_physiology_dataset(
        candidate_path, expected_records=len(config["included_record_ids"])
    )
    if frame["record_id"].tolist() != config["included_record_ids"]:
        raise ValueError("Candidate dataset participants differ from analysis config")
    source = _read_json(settings.root / config["source_manifest"])
    labels = {
        row["record_id"]: row["label"]
        for row in source["records"]
    }
    expected_labels = [
        labels[record_id] for record_id in config["included_record_ids"]
    ]
    if frame["label"].tolist() != expected_labels:
        raise ValueError("Candidate dataset labels differ from source manifest")
    model_directory = settings.root / "results" / "models" / analysis_id
    result = evaluate_primary_models(
        frame,
        model_directory,
        folds=int(config["primary_model"]["folds"]),
        random_seed=int(config["primary_model"]["random_seed"]),
        trees=int(config["primary_model"]["trees"]),
        expected_class_counts=config["expected_class_counts"],
        analysis_id=analysis_id,
    )
    result["reproducibility"] = {
        "analysis_cohort_config_sha256": dataset_digest(config_path),
        "preflight_report_sha256": dataset_digest(report_path),
        "subject_feature_dataset_sha256": dataset_digest(candidate_path),
        "fold_assignments_sha256": dataset_digest(
            model_directory / "fold_assignments.csv"
        ),
        "feature_missingness_by_modality": preflight[
            "feature_missingness_by_modality"
        ],
        "window_rows_observed": preflight["window_rows_observed"],
        "expected_channel_window_observations": preflight[
            "expected_channel_window_observations"
        ],
    }
    (model_directory / "metrics.json").write_text(
        json.dumps(_json_safe(result), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return _json_safe(result)
