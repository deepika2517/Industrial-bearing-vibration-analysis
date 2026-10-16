"""Time-domain statistical indicators for vibration signals."""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

try:
    from scipy.stats import kurtosis as scipy_kurtosis, skew as scipy_skew
except ImportError:  # pragma: no cover
    scipy_kurtosis = scipy_skew = None


def rms(signal: np.ndarray) -> float:
    """Root mean square — overall vibration energy."""
    return float(np.sqrt(np.mean(np.square(signal))))


def peak(signal: np.ndarray) -> float:
    """Maximum absolute amplitude."""
    return float(np.max(np.abs(signal)))


def peak_to_peak(signal: np.ndarray) -> float:
    """Difference between the maximum and minimum amplitude."""
    signal = np.asarray(signal, dtype=float)
    return float(signal.max() - signal.min())


def mean(signal: np.ndarray) -> float:
    return float(np.mean(signal))


def std(signal: np.ndarray) -> float:
    return float(np.std(signal))


def variance(signal: np.ndarray) -> float:
    return float(np.var(signal))


def skewness(signal: np.ndarray) -> float:
    """Third standardised moment — asymmetry of the amplitude distribution."""
    if scipy_skew is not None:
        return float(scipy_skew(signal))
    signal = np.asarray(signal, dtype=float)
    s = signal.std()
    return float(np.mean(((signal - signal.mean()) / s) ** 3)) if s else 0.0


def kurtosis(signal: np.ndarray, fisher: bool = True) -> float:
    """Fourth standardised moment — sensitivity to impulsive faults."""
    if scipy_kurtosis is not None:
        return float(scipy_kurtosis(signal, fisher=fisher))
    signal = np.asarray(signal, dtype=float)
    s = signal.std()
    k = float(np.mean(((signal - signal.mean()) / s) ** 4)) if s else 0.0
    return k - 3.0 if fisher else k


def crest_factor(signal: np.ndarray) -> float:
    """Peak divided by RMS — indicates impulsive content."""
    r = rms(signal)
    return float(peak(signal) / r) if r else 0.0


def shape_factor(signal: np.ndarray) -> float:
    """RMS divided by the mean absolute amplitude."""
    m = float(np.mean(np.abs(signal)))
    return float(rms(signal) / m) if m else 0.0


def impulse_factor(signal: np.ndarray) -> float:
    """Peak divided by the mean absolute amplitude."""
    m = float(np.mean(np.abs(signal)))
    return float(peak(signal) / m) if m else 0.0


def time_domain_features(signal: np.ndarray) -> Dict[str, float]:
    """Return all time-domain indicators as a dictionary."""
    return {
        "mean": mean(signal),
        "std": std(signal),
        "variance": variance(signal),
        "rms": rms(signal),
        "peak": peak(signal),
        "peak_to_peak": peak_to_peak(signal),
        "skewness": skewness(signal),
        "kurtosis": kurtosis(signal),
        "crest_factor": crest_factor(signal),
        "shape_factor": shape_factor(signal),
        "impulse_factor": impulse_factor(signal),
    }


def features_table(signals: Dict[str, np.ndarray]) -> pd.DataFrame:
    """Compute a feature table (rows = signals, columns = indicators)."""
    rows = {name: time_domain_features(sig) for name, sig in signals.items()}
    return pd.DataFrame.from_dict(rows, orient="index")
