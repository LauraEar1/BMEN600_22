from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tomllib


@dataclass(frozen=True)
class Settings:
    root: Path
    manifest_path: Path
    base_url: str
    expected_records: int
    labels: tuple[str, ...]
    minimum_free_gb: float
    timeout_seconds: int
    epoch_seconds: int
    chunk_seconds: int
    filter_edge_seconds: int
    filter_order: int
    flat_std_threshold: float
    maximum_nonfinite_fraction: float
    maximum_flat_window_fraction: float
    maximum_saturated_fraction: float
    saturation_limits: dict[str, float]
    filters: dict[str, tuple[float, float]]
    folds: int
    random_seed: int
    trees: int


def load_settings(config_path: Path | None = None) -> Settings:
    path = (config_path or Path("config/pipeline.toml")).resolve()
    root = path.parent.parent
    with path.open("rb") as stream:
        raw = tomllib.load(stream)
    processing = raw["processing"]
    filters = {
        modality: (float(values["low_hz"]), float(values["high_hz"]))
        for modality, values in raw["filters"].items()
    }
    return Settings(
        root=root,
        manifest_path=root / raw["cohort"]["manifest"],
        base_url=raw["download"]["base_url"].rstrip("/"),
        expected_records=int(raw["cohort"]["expected_records"]),
        labels=tuple(raw["cohort"]["labels"]),
        minimum_free_gb=float(raw["download"]["minimum_free_gb"]),
        timeout_seconds=int(raw["download"]["timeout_seconds"]),
        epoch_seconds=int(processing["epoch_seconds"]),
        chunk_seconds=int(processing["chunk_seconds"]),
        filter_edge_seconds=int(processing["filter_edge_seconds"]),
        filter_order=int(processing["filter_order"]),
        flat_std_threshold=float(processing["flat_std_threshold"]),
        maximum_nonfinite_fraction=float(processing["maximum_nonfinite_fraction"]),
        maximum_flat_window_fraction=float(processing["maximum_flat_window_fraction"]),
        maximum_saturated_fraction=float(processing["maximum_saturated_fraction"]),
        saturation_limits={
            "eeg": float(processing["eeg_saturation_uv"]),
            "eog": float(processing["eog_saturation_uv"]),
            "emg": float(processing["emg_saturation_uv"]),
            "ecg": float(processing["ecg_saturation_mv"]),
        },
        filters=filters,
        folds=int(raw["model"]["folds"]),
        random_seed=int(raw["model"]["random_seed"]),
        trees=int(raw["model"]["trees"]),
    )
