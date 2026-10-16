"""Signal preprocessing: detrending, filtering, normalisation, segmentation."""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

try:
    from scipy.signal import butter, detrend as scipy_detrend, filtfilt
except ImportError:  # pragma: no cover
    butter = filtfilt = None
    scipy_detrend = None


def remove_dc(signal: np.ndarray) -> np.ndarray:
    """Subtract the mean so the signal is zero-centred."""
    signal = np.asarray(signal, dtype=float)
    return signal - signal.mean()


def detrend(signal: np.ndarray, order: int = 1) -> np.ndarray:
    """Remove a linear (order=1) or constant (order=0) trend."""
    if scipy_detrend is not None:
        return scipy_detrend(np.asarray(signal, dtype=float), type="linear" if order else "constant")
    signal = np.asarray(signal, dtype=float)
    if order == 0:
        return signal - signal.mean()
    x = np.arange(len(signal))
    coeffs = np.polyfit(x, signal, order)
    return signal - np.polyval(coeffs, x)


def normalize(signal: np.ndarray, method: str = "zscore") -> np.ndarray:
    """Scale a signal to unit variance (``zscore``) or unit peak (``peak``)."""
    signal = np.asarray(signal, dtype=float)
    if method == "zscore":
        std = signal.std()
        return (signal - signal.mean()) / std if std else signal - signal.mean()
    if method == "peak":
        peak = np.max(np.abs(signal))
        return signal / peak if peak else signal
    raise ValueError(f"Unknown normalisation method: {method}")


def bandpass_filter(signal: np.ndarray, fs: float, low: float, high: float,
                    order: int = 4) -> np.ndarray:
    """Zero-phase Butterworth band-pass filter."""
    if butter is None or filtfilt is None:  # pragma: no cover
        raise ImportError("scipy is required for filtering")
    nyquist = 0.5 * fs
    low_n = max(low / nyquist, 1e-6)
    high_n = min(high / nyquist, 0.999)
    b, a = butter(order, [low_n, high_n], btype="band")
    return filtfilt(b, a, np.asarray(signal, dtype=float))


def segment_signal(signal: np.ndarray, window: int,
                   overlap: float = 0.5) -> List[np.ndarray]:
    """Split a signal into (optionally overlapping) windows."""
    signal = np.asarray(signal, dtype=float)
    if window <= 0:
        raise ValueError("window must be positive")
    step = max(int(window * (1 - overlap)), 1)
    return [signal[i:i + window] for i in range(0, len(signal) - window + 1, step)]


def preprocess_pipeline(signal: np.ndarray, fs: float,
                        band: Tuple[float, float] = (20.0, 2000.0),
                        normalize_method: str | None = "zscore") -> np.ndarray:
    """Standard preprocessing chain: detrend -> band-pass -> normalise."""
    out = detrend(remove_dc(signal), order=1)
    low, high = band
    if 0 < low < high < fs / 2:
        out = bandpass_filter(out, fs, low, high)
    if normalize_method:
        out = normalize(out, normalize_method)
    return out
