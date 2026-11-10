from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd


IDENTIFIER_COLUMNS = {"record_id", "subject_id", "label", "qc_status"}
MODALITY_PREFIXES = ("eeg__", "eog__", "emg__", "ecg__")
EXPECTED_CLASS_COUNTS = {"control": 16, "nfle": 40, "rbd": 22, "plm": 10}


def load_physiology_dataset(path: Path, expected_records: int = 88) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"record_id", "subject_id", "label"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Physiology dataset missing columns: {sorted(required - set(frame.columns))}")
    if frame["record_id"].duplicated().any() or frame["subject_id"].duplicated().any():
        raise ValueError("Physiology dataset must have one row per unique recording and subject")
    if len(frame) != expected_records:
        raise ValueError(f"Expected {expected_records} rows; found {len(frame)}")
    features = feature_columns(frame, MODALITY_PREFIXES)
    if not any(column.startswith("eeg__") for column in features):
        raise ValueError("Physiology dataset contains no EEG features")
    nonnumeric = [column for column in features if not pd.api.types.is_numeric_dtype(frame[column])]
    if nonnumeric:
        raise ValueError(f"Nonnumeric physiology features: {nonnumeric}")
    values = frame[features].to_numpy(dtype=float)
    if np.isinf(values).any():
        raise ValueError("Physiology features contain infinite values")
    return frame


def feature_columns(frame: pd.DataFrame, prefixes: tuple[str, ...]) -> list[str]:
    excluded = IDENTIFIER_COLUMNS
    return [
        column for column in frame.columns
        if column not in excluded and column.startswith(prefixes)
    ]


def validate_modality_datasets(frame: pd.DataFrame) -> dict[str, list[str]]:
    groups = {
        "eeg": ("eeg__",),
        "eeg_eog_emg": ("eeg__", "eog__", "emg__"),
        "eeg_eog_emg_ecg": MODALITY_PREFIXES,
    }
    results = {name: feature_columns(frame, prefixes) for name, prefixes in groups.items()}
    if any(not columns for columns in results.values()):
        raise ValueError("At least one requested modality model has no feature columns")
    for model_name, columns in results.items():
        for modality in groups[model_name]:
            modality_columns = [
                column for column in columns if column.startswith(modality)
            ]
            if not modality_columns or frame[modality_columns].isna().all(axis=None):
                raise ValueError(
                    f"Model {model_name} has no usable signal features for {modality[:-2]}"
                )
    row_ids = frame["record_id"].tolist()
    counts = frame["label"].value_counts().to_dict()
    if len(row_ids) != 88 or counts != EXPECTED_CLASS_COUNTS:
        raise ValueError(
            "All primary models require the verified 88-record cohort and class counts "
            f"{EXPECTED_CLASS_COUNTS}; found {counts}"
        )
    return results


def dataset_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
