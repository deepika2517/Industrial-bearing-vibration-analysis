"""Compare vibration signals across bearing conditions."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

from . import fft_analysis, statistical_analysis

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def _as_signal_dict(signals: Dict[str, object]) -> Dict[str, np.ndarray]:
    """Accept either raw arrays or ``VibrationSignal`` objects."""
    out: Dict[str, np.ndarray] = {}
    for name, obj in signals.items():
        out[name] = getattr(obj, "signal", obj)
    return out


def build_comparison_table(signals: Dict[str, object], fs: float) -> pd.DataFrame:
    """Combine time-domain and frequency-domain features into one table."""
    arrays = _as_signal_dict(signals)
    time_df = statistical_analysis.features_table(arrays)
    freq_df = fft_analysis.spectrum_table(arrays, fs)
    return time_df.join(freq_df)


def rank_conditions(table: pd.DataFrame, metric: str = "kurtosis") -> pd.Series:
    """Rank conditions by a metric (descending = more severe)."""
    if metric not in table.columns:
        raise KeyError(f"{metric} not in comparison table")
    return table[metric].sort_values(ascending=False)


def plot_comparison(signals: Dict[str, object], fs: float,
                    metric: str = "rms", save_path: Optional[str | Path] = None):
    """Bar chart comparing a single metric across conditions."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    table = build_comparison_table(signals, fs)
    fig, ax = plt.subplots(figsize=(8, 4))
    table[metric].plot(kind="bar", ax=ax, color="steelblue")
    ax.set_title(f"Bearing comparison — {metric}")
    ax.set_ylabel(metric)
    ax.set_xlabel("condition")
    fig.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
    return fig, ax


def plot_spectra(signals: Dict[str, object], fs: float,
                 save_path: Optional[str | Path] = None):
    """Overlay amplitude spectra of all conditions."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    arrays = _as_signal_dict(signals)
    fig, ax = plt.subplots(figsize=(9, 4))
    for name, sig in arrays.items():
        freqs, mag = fft_analysis.compute_fft(sig, fs)
        ax.plot(freqs, mag, linewidth=0.8, label=name)
    ax.set_xlim(0, fs / 2)
    ax.set_xlabel("frequency (Hz)")
    ax.set_ylabel("amplitude")
    ax.set_title("Amplitude spectra by condition")
    ax.legend(fontsize=8)
    fig.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
    return fig, ax


def save_comparison(table: pd.DataFrame, path: Optional[str | Path] = None) -> Path:
    """Write the comparison table to CSV under ``results/``."""
    path = Path(path) if path else RESULTS_DIR / "comparison_table.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path)
    return path
