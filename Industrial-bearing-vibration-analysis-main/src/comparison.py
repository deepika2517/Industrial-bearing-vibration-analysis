"""Compare vibration signals across bearing conditions."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

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


def aggregate_by_condition(table: pd.DataFrame, conditions: Dict[str, str],
                           order: Optional[List[str]] = None) -> pd.DataFrame:
    """Average the per-recording feature table across recordings per condition."""
    if not conditions:
        return table
    labels = [conditions.get(name, "unknown") for name in table.index]
    aggregated = table.copy()
    aggregated.insert(0, "condition", labels)
    aggregated = aggregated.groupby("condition", sort=False).mean(numeric_only=True)
    if order:
        existing = [c for c in order if c in aggregated.index]
        remaining = [c for c in aggregated.index if c not in existing]
        aggregated = aggregated.loc[existing + remaining]
    elif "healthy" in aggregated.index:
        others = [c for c in aggregated.index if c != "healthy"]
        aggregated = aggregated.loc[["healthy"] + others]
    return aggregated.round(4)


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


def plot_waveforms(signals: Dict[str, object], fs: float, max_samples: int = 2000,
                   save_path: Optional[str | Path] = None):
    """Grid of raw vibration waveforms, one panel per condition."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    arrays = _as_signal_dict(signals)
    names = list(arrays)
    cols = 2
    rows = max(1, -(-len(names) // cols))
    fig, axes = plt.subplots(rows, cols, figsize=(10, 2.4 * rows), squeeze=False)
    for ax, name in zip(axes.ravel(), names):
        sig = np.asarray(arrays[name], dtype=float)
        count = min(max_samples, len(sig))
        t = np.arange(count) / fs
        ax.plot(t, sig[:count], linewidth=0.7, color="steelblue")
        ax.set_title(name)
        ax.set_xlabel("time (s)")
        ax.set_ylabel("amplitude")
    for ax in axes.ravel()[len(names):]:
        ax.axis("off")
    fig.suptitle("Raw vibration waveforms by condition")
    fig.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
    return fig, axes


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
