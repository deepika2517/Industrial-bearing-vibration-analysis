"""Frequency-domain (FFT) analysis for vibration signals."""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd


def compute_fft(signal: np.ndarray, fs: float,
                window: str | None = "hann") -> Tuple[np.ndarray, np.ndarray]:
    """Return ``(frequencies, magnitude)`` of the single-sided amplitude spectrum."""
    signal = np.asarray(signal, dtype=float)
    n = len(signal)
    if window == "hann":
        signal = signal * np.hanning(n)
    spectrum = np.fft.rfft(signal)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    magnitude = np.abs(spectrum) * (2.0 / n)
    return freqs, magnitude


def power_spectrum(signal: np.ndarray, fs: float) -> Tuple[np.ndarray, np.ndarray]:
    """Return ``(frequencies, power)`` where power is magnitude squared."""
    freqs, mag = compute_fft(signal, fs)
    return freqs, mag ** 2


def dominant_frequencies(signal: np.ndarray, fs: float,
                         top_n: int = 5, min_freq: float = 1.0) -> np.ndarray:
    """Return the ``top_n`` dominant frequencies (Hz) above ``min_freq``."""
    freqs, mag = compute_fft(signal, fs)
    mask = freqs >= min_freq
    freqs, mag = freqs[mask], mag[mask]
    if len(freqs) == 0:
        return np.array([])
    idx = np.argsort(mag)[::-1][:top_n]
    return freqs[idx]


def spectral_centroid(signal: np.ndarray, fs: float) -> float:
    """Amplitude-weighted mean frequency — brightness of the spectrum."""
    freqs, mag = compute_fft(signal, fs)
    total = mag.sum()
    return float(np.sum(freqs * mag) / total) if total else 0.0


def spectral_bandwidth(signal: np.ndarray, fs: float) -> float:
    """Amplitude-weighted standard deviation around the spectral centroid."""
    freqs, mag = compute_fft(signal, fs)
    total = mag.sum()
    if not total:
        return 0.0
    centroid = spectral_centroid(signal, fs)
    return float(np.sqrt(np.sum(((freqs - centroid) ** 2) * mag) / total))


def band_energy_ratio(signal: np.ndarray, fs: float,
                      low: float, high: float) -> float:
    """Fraction of total spectral energy inside ``[low, high]`` Hz."""
    freqs, power = power_spectrum(signal, fs)
    total = power.sum()
    if not total:
        return 0.0
    mask = (freqs >= low) & (freqs <= high)
    return float(power[mask].sum() / total)


def frequency_features(signal: np.ndarray, fs: float,
                       bands: Dict[str, Tuple[float, float]] | None = None) -> Dict[str, float]:
    """Return dominant frequencies and band energy ratios."""
    dominant = dominant_frequencies(signal, fs, top_n=3)
    features: Dict[str, float] = {
        "spectral_centroid": spectral_centroid(signal, fs),
        "spectral_bandwidth": spectral_bandwidth(signal, fs),
    }
    for i, freq in enumerate(dominant, start=1):
        features[f"dominant_freq_{i}"] = float(freq)
    bands = bands or {"low": (0, 500), "mid": (500, 2000), "high": (2000, fs / 2)}
    for label, (low, high) in bands.items():
        features[f"band_energy_{label}"] = band_energy_ratio(signal, fs, low, high)
    return features


def spectrum_table(signals: Dict[str, np.ndarray], fs: float) -> pd.DataFrame:
    """Compute frequency-domain features for several signals."""
    rows = {name: frequency_features(sig, fs) for name, sig in signals.items()}
    return pd.DataFrame.from_dict(rows, orient="index")
