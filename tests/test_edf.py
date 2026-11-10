from pathlib import Path

import numpy as np
import pytest

from capsleep.config import load_settings
from capsleep import edf
from capsleep.edf import classify_channel, extract_recording, read_edf_header


def _edf_header(path: Path):
    channels = [
        ("EEG F3-C3", "uV", 128),
        ("ROC-LOC", "uV", 64),
        ("EMG1-EMG2", "uV", 128),
        ("ECG1-ECG2", "mV", 256),
    ]
    count = len(channels)
    fixed = (
        "0".ljust(8) + "patient".ljust(80) + "recording".ljust(80)
        + "01.01.09" + "22.06.03" + f"{256 + 256 * count:<8}"
        + "EDF+C".ljust(44) + f"{2:<8}" + f"{1:<8}" + f"{count:<4}"
    ).encode("ascii")
    signal_fields = [
        [name.ljust(16) for name, _, _ in channels],
        ["".ljust(80) for _ in channels],
        [unit.ljust(8) for _, unit, _ in channels],
        ["-100".ljust(8) for _ in channels],
        ["100".ljust(8) for _ in channels],
        ["-32768".ljust(8) for _ in channels],
        ["32767".ljust(8) for _ in channels],
        ["".ljust(80) for _ in channels],
        [str(samples).ljust(8) for _, _, samples in channels],
        ["".ljust(32) for _ in channels],
    ]
    body = "".join("".join(field) for field in signal_fields).encode("ascii")
    path.write_bytes(fixed + body)


def test_edf_header_preserves_each_native_sampling_rate_and_unit(tmp_path):
    path = tmp_path / "synthetic.edf"
    _edf_header(path)
    duration, record_duration, channels = read_edf_header(path)
    assert duration == 2
    assert record_duration == 1
    assert channels == [
        ("EEG F3-C3", "uV", 128.0),
        ("ROC-LOC", "uV", 64.0),
        ("EMG1-EMG2", "uV", 128.0),
        ("ECG1-ECG2", "mV", 256.0),
    ]


def test_edf_unknown_record_count_is_derived_from_file_size(tmp_path):
    path = tmp_path / "unknown-record-count.edf"
    _edf_header(path)
    header = bytearray(path.read_bytes())
    header[236:244] = "-1".ljust(8).encode("ascii")
    record_bytes = 2 * sum((128, 64, 128, 256))
    path.write_bytes(bytes(header) + bytes(record_bytes * 3))
    duration, _, _ = read_edf_header(path)
    assert duration == 3


def test_channel_classification_is_modality_specific():
    assert classify_channel("F2-F4") == "eeg"
    assert classify_channel("ROC-LOC") == "eog"
    assert classify_channel("EMG1-EMG2") == "emg"
    assert classify_channel("ECG1-ECG2") == "ecg"
    assert classify_channel("SpO2") is None


def _write_signal_edf(
    path: Path,
    missing_modalities=(),
    add_artifact_channels=False,
    duplicate_emg_labels=False,
    unitless_modalities=(),
):
    channels = [
        ("F3-C3", "uV", 128, 10.0, 40.0),
        ("ROC-LOC", "uV", 64, 1.0, 80.0),
        ("EMG1-EMG2", "uV", 128, 20.0, 120.0),
        ("ECG1-ECG2", "mV", 256, 2.0, 2.0),
        ("SpO2", "%", 1, 0.1, 90.0),
    ]
    channels = [
        channel for channel in channels
        if classify_channel(channel[0]) not in missing_modalities
    ]
    if add_artifact_channels:
        channels.extend([
            ("F4-C4", "uV", 128, 12.0, 10.0),
            ("EMG-flat", "uV", 128, 0.0, 40.0),
        ])
    if duplicate_emg_labels:
        channels.extend([
            ("CHIN", "uV", 128, 20.0, 120.0),
            ("CHIN", "uV", 128, 25.0, 120.0),
        ])
    count = len(channels)
    header_bytes = 256 + 256 * count
    fixed = (
        "0".ljust(8) + "patient".ljust(80) + "recording".ljust(80)
        + "01.01.09" + "22.06.03" + f"{header_bytes:<8}"
        + "EDF+C".ljust(44) + f"{60:<8}" + f"{1:<8}" + f"{count:<4}"
    ).encode("ascii")
    fields = [
        [name.ljust(16) for name, *_ in channels],
        ["".ljust(80) for _ in channels],
        [
            (
                "" if classify_channel(name) in unitless_modalities else unit
            ).ljust(8)
            for name, unit, *_ in channels
        ],
        [str(-limit).ljust(8) for _, _, _, _, limit in channels],
        [str(limit).ljust(8) for _, _, _, _, limit in channels],
        ["-32768".ljust(8) for _ in channels],
        ["32767".ljust(8) for _ in channels],
        ["".ljust(80) for _ in channels],
        [str(samples).ljust(8) for _, _, samples, *_ in channels],
        ["".ljust(32) for _ in channels],
    ]
    header = fixed + "".join("".join(field) for field in fields).encode("ascii")
    with path.open("wb") as stream:
        stream.write(header)
        for record in range(60):
            for _, _, sfreq, frequency, limit in channels:
                time = (np.arange(sfreq) + record * sfreq) / sfreq
                physical = limit * 0.2 * np.sin(2 * np.pi * frequency * time)
                digital = np.rint(physical / limit * 32767).astype("<i2")
                stream.write(digital.tobytes())


def test_windowed_extraction_uses_native_sampling_and_emits_qc(tmp_path):
    path = tmp_path / "synthetic.edf"
    _write_signal_edf(path)
    settings = load_settings()
    features, qc, channels, windows = extract_recording(
        path, "synthetic", "control", settings
    )
    native_rates = {row["channel"]: row["native_sfreq_hz"] for row in channels}
    assert native_rates == {
        "F3-C3": 128.0,
        "ROC-LOC": 64.0,
        "EMG1-EMG2": 128.0,
        "ECG1-ECG2": 256.0,
        "SpO2": 1.0,
    }
    assert qc["status"] == "complete"
    assert qc["missing_modalities"] == ""
    assert any(column.startswith("eeg__F3_C3__") for column in features)
    assert all(row["usable_window_count"] == 2 for row in channels if row["modality"])
    assert any(row["modality"] == "" for row in channels)
    assert len(windows) == 8
    eeg_window = next(row for row in windows if row["channel"] == "F3-C3")
    assert eeg_window["feature__dominant_frequency_hz"] == pytest.approx(10, abs=1)
    assert eeg_window["feature__rms"] == pytest.approx(8 / np.sqrt(2), rel=0.03)
    ecg_window = next(row for row in windows if row["channel"] == "ECG1-ECG2")
    assert ecg_window["physical_unit"] == "mV"
    assert ecg_window["feature__rms"] == pytest.approx(0.4 / np.sqrt(2), rel=0.03)


def test_missing_modalities_are_recorded_without_dropping_eeg(tmp_path):
    path = tmp_path / "missing-modalities.edf"
    _write_signal_edf(path, missing_modalities={"eog", "emg", "ecg"})
    settings = load_settings()
    features, qc, channels, windows = extract_recording(
        path, "missing", "control", settings
    )
    assert qc["status"] == "complete"
    assert qc["missing_modalities"] == "emg|eog|ecg"
    assert any(name.startswith("eeg__") for name in features)
    assert {row["modality"] for row in windows} == {"eeg"}
    assert {row["modality"] for row in channels} == {"", "eeg"}


def test_missing_physical_units_are_excluded_and_block_when_no_eeg_remains(tmp_path):
    path = tmp_path / "unitless-eeg.edf"
    _write_signal_edf(path, unitless_modalities={"eeg"})
    settings = load_settings()

    features, qc, channels, windows = extract_recording(
        path, "unitless", "control", settings
    )

    assert qc["status"] == "failed"
    assert qc["unverified_unit_channels"] == "F3-C3"
    assert qc["failure_reason"] == (
        "unitless: no EEG channels passed signal and physical-unit QC"
    )
    assert not any(name.startswith("eeg__") for name in features)
    assert windows
    eeg_row = next(row for row in channels if row["channel"] == "F3-C3")
    assert eeg_row["excluded"]
    assert eeg_row["exclusion_reasons"] == "missing_or_unsupported_physical_unit"


def test_saturation_and_flat_signal_quality_flags_are_reported(tmp_path):
    path = tmp_path / "artifacts.edf"
    _write_signal_edf(path, add_artifact_channels=True)
    settings = load_settings()
    settings.saturation_limits["eeg"] = 5.0
    features, qc, channels, windows = extract_recording(
        path, "artifacts", "control", settings
    )
    eeg_rows = {row["channel"]: row for row in channels if row["modality"] == "eeg"}
    assert eeg_rows["F3-C3"]["excluded"]
    assert "excess_saturation" in eeg_rows["F3-C3"]["exclusion_reasons"]
    assert not eeg_rows["F4-C4"]["excluded"]
    flat_rows = [row for row in windows if row["channel"] == "EMG-flat"]
    assert flat_rows and all(row["qc_is_flat"] for row in flat_rows)
    assert all(row["qc_channel_excluded"] for row in flat_rows)
    assert qc["status"] == "complete"
    assert any(name.startswith("eeg__F4_C4__") for name in features)


def test_nonfinite_samples_are_flagged_without_dropping_other_eeg_channels(
    tmp_path, monkeypatch
):
    path = tmp_path / "nonfinite.edf"
    _write_signal_edf(
        path,
        missing_modalities={"eog", "emg", "ecg"},
        add_artifact_channels=True,
    )
    time = np.arange(60 * 128) / 128
    signals = {
        "F3-C3": 8e-6 * np.sin(2 * np.pi * 10 * time),
        "F4-C4": 2e-6 * np.sin(2 * np.pi * 12 * time),
        "EMG-flat": np.zeros_like(time),
    }
    signals["F3-C3"][:200] = np.nan
    opened_raws = []

    class FakeRaw:
        def __init__(self, names):
            self.info = {"sfreq": 128.0}
            self.ch_names = names
            self.closed = False
            opened_raws.append(self)

        def get_data(self, picks, start, stop):
            return np.asarray([signals[picks[0]][start:stop]])

        def close(self):
            self.closed = True

    monkeypatch.setattr(
        edf.mne.io, "read_raw_edf", lambda *args, **kwargs: FakeRaw(kwargs["include"])
    )
    settings = load_settings()
    features, _, channels, windows = extract_recording(
        path, "nonfinite", "control", settings
    )
    channel_qc = {row["channel"]: row for row in channels}
    assert channel_qc["F3-C3"]["excluded"]
    assert "excess_nonfinite_samples" in channel_qc["F3-C3"]["exclusion_reasons"]
    assert not channel_qc["F4-C4"]["excluded"]
    assert any(name.startswith("eeg__F4_C4__") for name in features)
    f3_window = next(row for row in windows if row["channel"] == "F3-C3")
    assert f3_window["qc_nonfinite_fraction"] > 0
    assert all(raw.closed for raw in opened_raws)


def test_duplicate_edf_labels_are_mapped_to_unique_mne_and_feature_names(tmp_path):
    path = tmp_path / "duplicate-channel-labels.edf"
    _write_signal_edf(path, duplicate_emg_labels=True)
    settings = load_settings()
    _, _, channels, windows = extract_recording(
        path, "duplicates", "control", settings
    )
    chin_rows = [row for row in channels if row["channel"] == "CHIN"]
    assert [row["feature_channel"] for row in chin_rows] == ["CHIN", "CHIN__2"]
    assert {row["channel"] for row in windows if row["channel"].startswith("CHIN")} == {
        "CHIN", "CHIN__2"
    }
