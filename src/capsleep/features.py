from __future__ import annotations

import numpy as np
from scipy.signal import butter, find_peaks, welch, sosfiltfilt
from scipy.stats import kurtosis, skew


FEATURE_UNITS = {
    "mean": "normalized_signal_unit",
    "std": "normalized_signal_unit",
    "rms": "normalized_signal_unit",
    "line_length_per_second": "normalized_signal_unit/second",
    "skew": "1",
    "kurtosis": "1",
    "zero_crossings_per_second": "1/second",
    "hjorth_activity": "normalized_signal_unit^2",
    "hjorth_mobility": "Hz",
    "hjorth_complexity": "1",
    "spectral_entropy": "1",
    "dominant_frequency_hz": "Hz",
    "r_peak_count": "count/window",
    "heart_rate_bpm": "beats/minute",
    "rr_sdnn_ms": "ms",
}


FEATURE_CALCULATIONS = {
    "mean": "Arithmetic mean of the filtered window samples.",
    "std": "Population standard deviation of the filtered window samples.",
    "rms": "Square root of the mean squared filtered window samples.",
    "line_length_per_second": "Mean absolute first difference multiplied by native sample rate.",
    "skew": "Bias-corrected Fisher-Pearson sample skewness.",
    "kurtosis": "Bias-corrected excess Fisher kurtosis.",
    "zero_crossings_per_second": "Sign changes around the centered mean divided by window duration.",
    "hjorth_activity": "Variance of the centered filtered signal.",
    "hjorth_mobility": "Square root of first-difference variance divided by signal variance; derivatives use native Hz.",
    "hjorth_complexity": "Second-derivative mobility divided by first-derivative mobility.",
    "spectral_entropy": "Shannon entropy of normalized Welch power bins in the modality feature band, divided by log(bin count).",
    "dominant_frequency_hz": "Frequency bin with maximum Welch power.",
    "r_peak_count": "Count of polarity-selected ECG peaks with 0.3-second minimum separation.",
    "heart_rate_bpm": "60 divided by the median 0.3-2.0-second inter-peak interval.",
    "rr_sdnn_ms": "Sample standard deviation of valid inter-peak intervals, converted to milliseconds.",
}


def feature_schema(modality: str) -> dict[str, dict[str, str]]:
    signal_unit = "mV" if modality == "ecg" else "uV"
    definitions = {}
    for name, unit in FEATURE_UNITS.items():
        if name.startswith(("r_peak_", "heart_rate_", "rr_sdnn_")) and modality != "ecg":
            continue
        resolved_unit = unit.replace("normalized_signal_unit", signal_unit)
        definitions[name] = {
            "unit": resolved_unit,
            "calculation": FEATURE_CALCULATIONS[name],
        }
    for band in _bands(modality):
        definitions[f"power_{band}"] = {
            "unit": f"{signal_unit}^2",
            "calculation": f"Welch power spectral density integrated over the {band} modality band.",
        }
        definitions[f"relative_power_{band}"] = {
            "unit": "1",
            "calculation": f"Power in the {band} modality band divided by total integrated power over all represented frequencies.",
        }
    return definitions


def filter_signal(
    signal: np.ndarray, sfreq: float, low_hz: float, high_hz: float, order: int
) -> np.ndarray:
    nyquist = sfreq / 2.0
    high = min(high_hz, nyquist * 0.95)
    if low_hz <= 0 or high <= low_hz or len(signal) < 16:
        raise ValueError(f"Cannot bandpass {low_hz}-{high_hz} Hz at {sfreq} Hz")
    sos = butter(order, [low_hz, high], btype="bandpass", fs=sfreq, output="sos")
    return sosfiltfilt(sos, signal)


def _bands(modality: str) -> dict[str, tuple[float, float]]:
    if modality == "eeg":
        return {
            "delta": (0.5, 4), "theta": (4, 8), "alpha": (8, 13),
            "sigma": (13, 16), "beta": (16, 30),
        }
    if modality == "eog":
        return {"slow": (0.1, 1), "low": (1, 5), "high": (5, 14)}
    if modality == "emg":
        return {"low": (10, 30), "mid": (30, 50), "high": (50, 95)}
    return {"low": (0.5, 5), "mid": (5, 15), "high": (15, 38)}


def channel_features(signal: np.ndarray, sfreq: float, modality: str) -> dict[str, float]:
    signal = np.asarray(signal, dtype=np.float64)
    if signal.ndim != 1 or len(signal) < 16 or not np.isfinite(signal).all():
        raise ValueError("Feature input must be a finite one-dimensional signal")
    centered = signal - np.mean(signal)
    variance = float(np.var(centered))
    values = {
        "mean": float(np.mean(signal)),
        "std": float(np.std(signal)),
        "rms": float(np.sqrt(np.mean(signal**2))),
        "line_length_per_second": float(np.mean(np.abs(np.diff(signal))) * sfreq),
        "skew": float(skew(signal, bias=False)) if variance > 0 else 0.0,
        "kurtosis": float(kurtosis(signal, fisher=True, bias=False)) if variance > 0 else 0.0,
        "zero_crossings_per_second": float(
            np.count_nonzero(np.diff(np.signbit(centered))) / (len(signal) / sfreq)
        ),
        "hjorth_activity": variance,
    }
    first = np.diff(signal) * sfreq
    mobility = float(np.sqrt(np.var(first) / variance)) if variance > 0 else 0.0
    second = np.diff(first) * sfreq
    derivative_mobility = (
        float(np.sqrt(np.var(second) / np.var(first))) if np.var(first) > 0 else 0.0
    )
    values["hjorth_mobility"] = mobility
    values["hjorth_complexity"] = (
        derivative_mobility / mobility if mobility > 0 else 0.0
    )
    frequencies, power = welch(signal, fs=sfreq, nperseg=min(len(signal), int(sfreq * 4)))
    total_power = float(np.trapezoid(power, frequencies))
    for name, (low, high) in _bands(modality).items():
        high = min(high, sfreq / 2.0)
        mask = (frequencies >= low) & (frequencies < high)
        band_power = float(np.trapezoid(power[mask], frequencies[mask])) if mask.sum() > 1 else 0.0
        values[f"power_{name}"] = band_power
        values[f"relative_power_{name}"] = band_power / total_power if total_power > 0 else 0.0
    if total_power > 0:
        bands = _bands(modality)
        entropy_low = min(low for low, _ in bands.values())
        entropy_high = min(max(high for _, high in bands.values()), sfreq / 2.0)
        entropy_mask = (frequencies >= entropy_low) & (frequencies < entropy_high)
        entropy_power = power[entropy_mask]
        normalized = entropy_power / max(
            float(entropy_power.sum()), np.finfo(float).tiny
        )
        values["spectral_entropy"] = float(
            -np.sum(normalized[normalized > 0] * np.log(normalized[normalized > 0]))
            / np.log(len(normalized))
            if len(normalized) > 1 and normalized.sum() > 0
            else 0.0
        )
        values["dominant_frequency_hz"] = float(frequencies[int(np.argmax(power))])
    else:
        values["spectral_entropy"] = 0.0
        values["dominant_frequency_hz"] = 0.0
    if modality == "ecg":
        distance = max(1, int(round(0.3 * sfreq)))
        prominence = max(float(np.std(signal)) * 0.5, np.finfo(float).eps)
        positive, positive_properties = find_peaks(
            signal, distance=distance, prominence=prominence
        )
        negative, negative_properties = find_peaks(
            -signal, distance=distance, prominence=prominence
        )
        if len(positive) > len(negative):
            peaks = positive
            prominences = positive_properties["prominences"]
        else:
            peaks = negative
            prominences = negative_properties["prominences"]
        intervals = np.diff(peaks) / sfreq
        valid_intervals = intervals[(intervals >= 0.3) & (intervals <= 2.0)]
        values["r_peak_count"] = float(len(peaks))
        values["heart_rate_bpm"] = (
            float(60.0 / np.median(valid_intervals))
            if len(valid_intervals) else float("nan")
        )
        values["rr_sdnn_ms"] = (
            float(np.std(valid_intervals, ddof=1) * 1000)
            if len(valid_intervals) > 1 else float("nan")
        )
    return values


def summarize_windows(windows: list[dict[str, float]]) -> dict[str, float]:
    if not windows:
        raise ValueError("Cannot summarize an empty set of valid signal windows")
    names = windows[0].keys()
    matrix = np.asarray([[window[name] for name in names] for window in windows])
    output: dict[str, float] = {}
    for index, name in enumerate(names):
        values = matrix[:, index]
        output[f"{name}_median"] = float(np.median(values))
        output[f"{name}_iqr"] = float(np.percentile(values, 75) - np.percentile(values, 25))
    return output
