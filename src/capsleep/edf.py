from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
import re

import mne
import numpy as np

from .config import Settings
from .features import channel_features, feature_schema, filter_signal, summarize_windows


@dataclass(frozen=True)
class EDFChannel:
    name: str
    feature_name: str
    physical_unit: str
    native_sfreq: float
    modality: str | None


def _ascii_field(data: bytes, start: int, width: int) -> str:
    return data[start : start + width].decode("ascii", errors="strict").strip()


def parse_edf_header(
    header: bytes, file_size: int | None = None
) -> tuple[float, float, list[tuple[str, str, float]]]:
    if len(header) < 256:
        raise ValueError("Truncated EDF fixed header")
    fixed = header[:256]
    if _ascii_field(fixed, 0, 8) not in {"0", "0.1"}:
        raise ValueError("Unsupported EDF version")
    header_bytes = int(_ascii_field(fixed, 184, 8))
    record_count = int(_ascii_field(fixed, 236, 8))
    record_duration = float(_ascii_field(fixed, 244, 8))
    signal_count = int(_ascii_field(fixed, 252, 4))
    if (
        not 0 < signal_count <= 4096
        or record_duration <= 0
        or record_count == 0
        or not 256 + signal_count * 256 <= header_bytes <= 256 + 4096 * 256
    ):
        raise ValueError("Invalid EDF dimensions")
    if len(header) < header_bytes:
        raise ValueError("Truncated EDF signal header")
    remainder = header[256:header_bytes]
    labels = [
        _ascii_field(remainder, index * 16, 16) for index in range(signal_count)
    ]
    unit_start = 16 * signal_count + 80 * signal_count
    units = [
        _ascii_field(remainder, unit_start + index * 8, 8)
        for index in range(signal_count)
    ]
    samples_start = 216 * signal_count
    samples = [
        int(_ascii_field(remainder, samples_start + index * 8, 8))
        for index in range(signal_count)
    ]
    if any(value <= 0 for value in samples):
        raise ValueError("Invalid samples per data record")
    if record_count == -1 and file_size is not None:
        record_bytes = 2 * sum(samples)
        payload_bytes = file_size - header_bytes
        if payload_bytes < 0 or payload_bytes % record_bytes:
            raise ValueError("EDF file size is inconsistent with its data-record layout")
        record_count = payload_bytes // record_bytes
        if record_count <= 0:
            raise ValueError("EDF file contains no complete data records")
    channels = [
        (name, unit, count / record_duration)
        for name, unit, count in zip(labels, units, samples, strict=True)
    ]
    duration = record_count * record_duration if record_count > 0 else float("nan")
    return duration, record_duration, channels


def read_edf_header(path: Path) -> tuple[float, float, list[tuple[str, str, float]]]:
    with path.open("rb") as stream:
        fixed = stream.read(256)
        if len(fixed) != 256:
            raise ValueError(f"Truncated EDF fixed header: {path}")
        header_bytes = int(_ascii_field(fixed, 184, 8))
        signal_count = int(_ascii_field(fixed, 252, 4))
        if not 0 < signal_count <= 4096 or not 256 + signal_count * 256 <= header_bytes <= 256 + 4096 * 256:
            raise ValueError(f"Invalid EDF header dimensions: {path}")
        remainder = stream.read(header_bytes - 256)
    return parse_edf_header(fixed + remainder, path.stat().st_size)


def classify_channel(name: str) -> str | None:
    normalized = re.sub(r"[^A-Z0-9]", "", name.upper())
    if re.search(r"ECG|EKG", normalized):
        return "ecg"
    if re.search(r"EMG|TIB|CHIN|DX\d|SX\d", normalized):
        return "emg"
    if re.search(r"EOG|ROC|LOC", normalized):
        return "eog"
    if "EEG" in normalized or re.match(r"^(FP|F|C|P|O|T)\d", normalized):
        return "eeg"
    return None


def inspect_channels(path: Path) -> tuple[float, list[EDFChannel]]:
    duration, _, raw_channels = read_edf_header(path)
    if not np.isfinite(duration):
        raw = mne.io.read_raw_edf(path, preload=False, infer_types=False, verbose="ERROR")
        duration = raw.n_times / raw.info["sfreq"]
        raw.close()
    seen_names: dict[str, int] = defaultdict(int)
    channels = []
    for name, unit, sfreq in raw_channels:
        seen_names[name] += 1
        feature_name = name if seen_names[name] == 1 else f"{name}__{seen_names[name]}"
        channels.append(
            EDFChannel(name, feature_name, unit, sfreq, classify_channel(name))
        )
    return duration, channels


def _signal_scale(modality: str) -> float:
    return 1000.0 if modality == "ecg" else 1_000_000.0


def _saturation_limit(modality: str, settings: Settings) -> float:
    return settings.saturation_limits[modality]


def _has_supported_signal_unit(unit: str) -> bool:
    normalized = unit.strip().lower().replace("μ", "µ")
    return normalized in {
        "v", "volt", "volts",
        "mv", "millivolt", "millivolts",
        "uv", "µv", "microvolt", "microvolts",
        "nv", "nanovolt", "nanovolts",
    }


def extract_recording(
    path: Path, record_id: str, label: str, settings: Settings
) -> tuple[dict, dict, list[dict], list[dict]]:
    duration, channels = inspect_channels(path)
    relevant = [channel for channel in channels if channel.modality]
    if not any(channel.modality == "eeg" for channel in relevant):
        raise ValueError(f"{record_id}: no recognized EEG channels")
    channel_rows: list[dict] = []
    qc_excluded: list[str] = []
    unverified_unit_channels = []
    for channel in channels:
        if channel.modality is None:
            channel_rows.append({
                "record_id": record_id,
                "channel": channel.name,
                "feature_channel": channel.feature_name,
                "modality": "",
                "physical_unit": channel.physical_unit,
                "native_sfreq_hz": channel.native_sfreq,
                "nonfinite_fraction": float("nan"),
                "flat_window_fraction": float("nan"),
                "saturated_fraction": float("nan"),
                "usable_window_count": 0,
                "excluded": False,
                "exclusion_reasons": "not_selected_modality",
            })
        elif not _has_supported_signal_unit(channel.physical_unit):
            qc_excluded.append(channel.name)
            unverified_unit_channels.append(channel.name)
            channel_rows.append({
                "record_id": record_id,
                "channel": channel.name,
                "feature_channel": channel.feature_name,
                "modality": channel.modality,
                "physical_unit": channel.physical_unit,
                "native_sfreq_hz": channel.native_sfreq,
                "nonfinite_fraction": float("nan"),
                "flat_window_fraction": float("nan"),
                "saturated_fraction": float("nan"),
                "usable_window_count": 0,
                "excluded": True,
                "exclusion_reasons": "missing_or_unsupported_physical_unit",
            })
    processable = [
        channel for channel in relevant
        if _has_supported_signal_unit(channel.physical_unit)
    ]
    grouped: dict[tuple[str, float], list[EDFChannel]] = defaultdict(list)
    for channel in processable:
        grouped[(channel.modality, channel.native_sfreq)].append(channel)
    features: dict[str, float] = {}
    window_rows: list[dict] = []
    for (modality, sfreq), group in grouped.items():
        names = list(dict.fromkeys(channel.name for channel in group))
        raw = mne.io.read_raw_edf(
            path, include=names, preload=False, infer_types=False,
            stim_channel=None, verbose="ERROR",
        )
        if len(raw.ch_names) != len(group):
            raw.close()
            raise ValueError(
                f"{record_id}: MNE returned {len(raw.ch_names)} channel names "
                f"for {len(group)} selected EDF channels: {names}"
            )
        mne_names = {}
        for channel, mne_name in zip(group, raw.ch_names, strict=True):
            if mne_name != channel.name and not re.fullmatch(
                rf"{re.escape(channel.name)}-\d+", mne_name
            ):
                raw.close()
                raise ValueError(
                    f"{record_id}: cannot map EDF channel {channel.name!r} "
                    f"to MNE channel {mne_name!r}"
                )
            mne_names[channel.feature_name] = mne_name
        if not np.isclose(raw.info["sfreq"], sfreq, rtol=0, atol=1e-8):
            raw.close()
            raise ValueError(
                f"{record_id}: MNE sampling rate {raw.info['sfreq']} does not "
                f"match EDF-native rate {sfreq} for {names}"
            )
        epoch_samples = int(round(settings.epoch_seconds * sfreq))
        guard = settings.filter_edge_seconds
        guard_samples = int(round(guard * sfreq))
        try:
          for channel in group:
            window_features: list[dict[str, float]] = []
            channel_window_rows: list[dict] = []
            total_samples = 0
            nonfinite_samples = 0
            flat_windows = 0
            total_windows = 0
            saturated_samples = 0
            for chunk_start in range(0, int(np.ceil(duration)), settings.chunk_seconds):
                chunk_end = min(chunk_start + settings.chunk_seconds, duration)
                read_start = max(0.0, chunk_start - guard)
                read_end = min(duration, chunk_end + guard)
                start_sample = int(round(read_start * sfreq))
                stop_sample = int(round(read_end * sfreq))
                data = raw.get_data(
                    picks=[mne_names[channel.feature_name]],
                    start=start_sample,
                    stop=stop_sample,
                )[0]
                nonfinite = ~np.isfinite(data)
                if nonfinite.any():
                    if nonfinite.all():
                        data.fill(0.0)
                    else:
                        good = np.flatnonzero(~nonfinite)
                        data[nonfinite] = np.interp(
                            np.flatnonzero(nonfinite), good, data[good]
                        )
                data *= _signal_scale(modality)
                left_pad = guard_samples if read_start == 0 else 0
                right_pad = guard_samples if read_end >= duration else 0
                padded = np.pad(data, (left_pad, right_pad), mode="reflect") if (
                    left_pad or right_pad
                ) else data
                nonfinite_padded = np.pad(
                    nonfinite, (left_pad, right_pad), mode="constant"
                )
                filtered = filter_signal(
                    padded, sfreq, *settings.filters[modality], settings.filter_order
                )
                crop_start = left_pad + int(round((chunk_start - read_start) * sfreq))
                usable_end = left_pad + int(round((chunk_end - read_start) * sfreq))
                usable = filtered[crop_start:usable_end]
                quality_usable = padded[crop_start:usable_end]
                nonfinite_usable = nonfinite_padded[crop_start:usable_end]
                for offset in range(0, len(usable) - epoch_samples + 1, epoch_samples):
                    epoch = usable[offset : offset + epoch_samples]
                    raw_epoch = quality_usable[offset : offset + epoch_samples]
                    epoch_nonfinite = nonfinite_usable[offset : offset + epoch_samples]
                    total_windows += 1
                    total_samples += len(raw_epoch)
                    nonfinite_samples += int(epoch_nonfinite.sum())
                    flat = float(np.std(raw_epoch)) <= settings.flat_std_threshold
                    flat_windows += int(flat)
                    epoch_saturated = int(
                        np.count_nonzero(
                            np.abs(raw_epoch) >= _saturation_limit(modality, settings)
                        )
                    )
                    saturated_samples += epoch_saturated
                    window_start_seconds = chunk_start + offset / sfreq
                    window_index = int(round(window_start_seconds / settings.epoch_seconds))
                    values = (
                        channel_features(epoch, sfreq, modality)
                        if not flat
                        else {}
                    )
                    if not flat:
                        window_features.append(values)
                    channel_window_rows.append({
                        "record_id": record_id,
                        "subject_id": record_id,
                        "label": label,
                        "window_index": window_index,
                        "window_start_seconds": window_index * settings.epoch_seconds,
                        "channel": channel.feature_name,
                        "modality": modality,
                        "physical_unit": channel.physical_unit,
                        "native_sfreq_hz": sfreq,
                        "qc_is_flat": flat,
                        "qc_nonfinite_fraction": float(epoch_nonfinite.mean()),
                        "qc_saturated_fraction": float(epoch_saturated / len(raw_epoch)),
                        **{
                            f"feature__{name}": value
                            for name, value in values.items()
                        },
                    })
            nonfinite_fraction = nonfinite_samples / max(total_samples, 1)
            flat_fraction = flat_windows / max(total_windows, 1)
            saturated_fraction = saturated_samples / max(total_windows * epoch_samples, 1)
            exclusions = []
            if nonfinite_fraction > settings.maximum_nonfinite_fraction:
                exclusions.append("excess_nonfinite_samples")
            if flat_fraction > settings.maximum_flat_window_fraction:
                exclusions.append("excess_flat_windows")
            if saturated_fraction > settings.maximum_saturated_fraction:
                exclusions.append("excess_saturation")
            if total_windows == 0 or not window_features:
                exclusions.append("no_complete_epochs")
            for window_row in channel_window_rows:
                window_row["qc_channel_excluded"] = bool(exclusions)
            window_rows.extend(channel_window_rows)
            if exclusions:
                qc_excluded.append(channel.name)
            else:
                prefix = (
                    f"{modality}__"
                    f"{re.sub(r'[^A-Za-z0-9]+', '_', channel.feature_name).strip('_')}"
                )
                features.update(
                    {f"{prefix}__{name}": value
                     for name, value in summarize_windows(window_features).items()}
                )
            channel_rows.append({
                "record_id": record_id,
                "channel": channel.name,
                "feature_channel": channel.feature_name,
                "modality": modality,
                "physical_unit": channel.physical_unit,
                "native_sfreq_hz": channel.native_sfreq,
                "nonfinite_fraction": nonfinite_fraction,
                "flat_window_fraction": flat_fraction,
                "saturated_fraction": saturated_fraction,
                "usable_window_count": len(window_features) if not exclusions else 0,
                "excluded": bool(exclusions),
                "exclusion_reasons": "|".join(exclusions),
            })
        finally:
            raw.close()
    has_usable_eeg = any(key.startswith("eeg__") for key in features)
    status = "complete" if has_usable_eeg else "failed"
    failure_reason = (
        "" if has_usable_eeg
        else f"{record_id}: no EEG channels passed signal and physical-unit QC"
    )
    row = {"record_id": record_id, "subject_id": record_id, "label": label, **features}
    qc = {
        "record_id": record_id,
        "label": label,
        "duration_seconds": duration,
        "channel_count": len(channels),
        "recognized_channel_count": len(relevant),
        "excluded_channel_count": len(qc_excluded),
        "missing_modalities": "|".join(
            modality for modality in ("eeg", "emg", "eog", "ecg")
            if not any(ch.modality == modality for ch in relevant)
        ),
        "qc_excluded_channels": "|".join(qc_excluded),
        "unverified_unit_channels": "|".join(unverified_unit_channels),
        "status": status,
    }
    if failure_reason:
        qc["failure_reason"] = failure_reason
    return row, qc, channel_rows, window_rows
