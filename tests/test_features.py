import numpy as np
import pytest

from capsleep.features import (
    channel_features,
    feature_schema,
    filter_signal,
    summarize_windows,
)


def test_filter_and_interpretable_features_are_finite():
    sfreq = 128.0
    time = np.arange(int(30 * sfreq)) / sfreq
    signal = 20 * np.sin(2 * np.pi * 10 * time) + 2 * np.sin(2 * np.pi * 50 * time)
    filtered = filter_signal(signal, sfreq, 0.3, 35, 4)
    features = channel_features(filtered, sfreq, "eeg")
    assert np.isfinite(list(features.values())).all()
    assert features["std"] > 10
    assert features["dominant_frequency_hz"] == pytest.approx(10, abs=1)
    assert features["relative_power_alpha"] > features["relative_power_beta"]


def test_filter_rejects_impossible_native_sampling_rate():
    with pytest.raises(ValueError, match="Cannot bandpass"):
        filter_signal(np.ones(100), 20, 10, 100, 4)


def test_known_frequency_and_amplitude_have_expected_spectral_features():
    sfreq = 128.0
    time = np.arange(int(30 * sfreq)) / sfreq
    signal = 10.0 * np.sin(2 * np.pi * 10 * time)
    features = channel_features(signal, sfreq, "eeg")
    assert features["rms"] == pytest.approx(10 / np.sqrt(2), rel=1e-3)
    assert features["dominant_frequency_hz"] == pytest.approx(10.0, abs=0.25)
    assert features["relative_power_alpha"] > 0.95
    assert features["relative_power_delta"] < 1e-6
    assert set(features) == set(feature_schema("eeg"))
    assert feature_schema("eeg")["rms"]["unit"] == "uV"


def test_modality_filter_suppresses_out_of_band_artifact_frequency():
    sfreq = 128.0
    time = np.arange(int(30 * sfreq)) / sfreq
    high_frequency_artifact = np.sin(2 * np.pi * 50 * time)
    filtered = filter_signal(high_frequency_artifact, sfreq, 0.3, 35, 4)
    assert np.std(filtered) < 0.05


def test_feature_definitions_cover_all_modality_specific_outputs():
    for modality in ("eeg", "eog", "emg", "ecg"):
        time = np.arange(30 * 128) / 128
        features = channel_features(
            np.sin(2 * np.pi * 10 * time), 128, modality
        )
        definitions = feature_schema(modality)
        assert set(features) == set(definitions)
        assert all(item["unit"] and item["calculation"] for item in definitions.values())
    assert feature_schema("ecg")["heart_rate_bpm"]["unit"] == "beats/minute"


def test_recording_summary_uses_window_median_and_iqr():
    summary = summarize_windows([{"rms": 2.0}, {"rms": 4.0}, {"rms": 8.0}])
    assert summary["rms_median"] == 4.0
    assert summary["rms_iqr"] == 3.0


def test_time_domain_features_are_comparable_across_native_sampling_rates():
    features = []
    for sfreq in (64.0, 128.0):
        time = np.arange(int(30 * sfreq)) / sfreq
        signal = np.sin(2 * np.pi * 10 * time)
        features.append(channel_features(signal, sfreq, "eeg"))
    assert features[0]["line_length_per_second"] == pytest.approx(
        features[1]["line_length_per_second"], rel=0.03
    )
    assert features[0]["hjorth_mobility"] == pytest.approx(
        features[1]["hjorth_mobility"], rel=0.03
    )
    assert features[0]["spectral_entropy"] == pytest.approx(
        features[1]["spectral_entropy"], rel=0.03
    )


def test_ecg_features_include_a_basic_r_peak_rate_estimate():
    sfreq = 256.0
    signal = np.zeros(int(30 * sfreq))
    for second in range(1, 29):
        signal[second * int(sfreq)] = 1.0
    features = channel_features(signal, sfreq, "ecg")
    assert features["heart_rate_bpm"] == pytest.approx(60.0)
    assert features["r_peak_count"] >= 25
