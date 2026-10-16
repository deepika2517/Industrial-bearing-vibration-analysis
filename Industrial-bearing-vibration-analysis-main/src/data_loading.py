"""Loading utilities for bearing vibration signals.

Supports CSV and MATLAB ``.mat`` recordings as well as a synthetic signal
generator so the pipeline can run without the full dataset.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd

try:  # scipy is optional for the synthetic-only path
    from scipy.io import loadmat
except ImportError:  # pragma: no cover
    loadmat = None


@dataclass
class VibrationSignal:
    """A single vibration recording.

    Attributes:
        name: Human-readable identifier (e.g. ``"inner_race_0.007"``).
        condition: Bearing health condition label (e.g. ``"healthy"``).
        signal: 1-D array of vibration amplitude samples.
        fs: Sampling frequency in Hz.
    """

    name: str
    condition: str
    signal: np.ndarray
    fs: float

    def __post_init__(self) -> None:
        self.signal = np.asarray(self.signal, dtype=float).ravel()

    @property
    def duration(self) -> float:
        """Recording length in seconds."""
        return len(self.signal) / self.fs


def load_csv(path: Union[str, Path], column: str = "amplitude", fs: float = 12000.0,
             name: Optional[str] = None, condition: str = "unknown") -> VibrationSignal:
    """Load a single-column (or named-column) CSV recording."""
    path = Path(path)
    frame = pd.read_csv(path)
    if column not in frame.columns:
        column = frame.columns[0]
    return VibrationSignal(
        name=name or path.stem,
        condition=condition,
        signal=frame[column].to_numpy(),
        fs=fs,
    )


def load_mat(path: Union[str, Path], key: Optional[str] = None, fs: float = 12000.0,
             name: Optional[str] = None, condition: str = "unknown") -> VibrationSignal:
    """Load a MATLAB ``.mat`` recording.

    Uses ``key`` when given, otherwise selects the first floating-point array.
    """
    if loadmat is None:  # pragma: no cover
        raise ImportError("scipy is required to read .mat files")
    path = Path(path)
    data = loadmat(path)
    candidates = [
        k for k, v in data.items()
        if not k.startswith("__") and isinstance(v, np.ndarray) and np.issubdtype(v.dtype, np.floating)
    ]
    if not candidates:
        raise ValueError(f"No numeric array found in {path}")
    selected = key or candidates[0]
    return VibrationSignal(
        name=name or f"{path.stem}_{selected}",
        condition=condition,
        signal=data[selected].ravel(),
        fs=fs,
    )


def load_directory(root: Union[str, Path], fs: float = 12000.0,
                   condition: Optional[str] = None) -> Dict[str, VibrationSignal]:
    """Load every CSV/``.mat`` file under ``root``.

    The condition defaults to the immediate parent folder name.
    """
    root = Path(root)
    signals: Dict[str, VibrationSignal] = {}
    for path in sorted(root.rglob("*")):
        if path.suffix.lower() not in {".csv", ".mat"}:
            continue
        cond = condition or path.parent.name
        try:
            if path.suffix.lower() == ".csv":
                sig = load_csv(path, fs=fs, condition=cond)
            else:
                sig = load_mat(path, fs=fs, condition=cond)
        except Exception as exc:  # noqa: BLE001 - keep loading the rest
            print(f"[data_loading] skipped {path}: {exc}")
            continue
        signals[sig.name] = sig
    return signals


def generate_synthetic_signal(condition: str = "healthy", fs: float = 12000.0,
                              duration: float = 1.0, seed: Optional[int] = 42) -> np.ndarray:
    """Generate a synthetic bearing vibration signal.

    Healthy bearings are dominated by a shaft frequency plus broadband noise;
    faulty conditions add a characteristic defect frequency and impulse train.
    """
    rng = np.random.default_rng(seed)
    n = int(fs * duration)
    t = np.arange(n) / fs

    shaft = 25.0  # Hz, typical shaft speed
    signal = 1.0 * np.sin(2 * np.pi * shaft * t)
    signal += 0.05 * rng.standard_normal(n)

    defect_freqs = {
        "healthy": None,
        "outer_race": 3.0 * shaft,
        "inner_race": 4.2 * shaft,
        "ball": 2.4 * shaft,
    }
    fault = defect_freqs.get(condition)
    if fault is not None:
        signal += 0.6 * np.sin(2 * np.pi * fault * t)
        impulses = np.zeros(n)
        step = max(int(fs / fault), 1)
        impulses[::step] = 1.0
        decay = np.exp(-np.arange(200) / 40.0)
        impulses = np.convolve(impulses, decay)[:n]
        signal += 0.8 * impulses
        signal += 0.15 * rng.standard_normal(n)

    return signal


def make_demo_dataset(fs: float = 12000.0, duration: float = 1.0) -> Dict[str, VibrationSignal]:
    """Build one synthetic signal per supported condition."""
    return {
        cond: VibrationSignal(
            name=cond,
            condition=cond,
            signal=generate_synthetic_signal(cond, fs=fs, duration=duration),
            fs=fs,
        )
        for cond in ("healthy", "outer_race", "inner_race", "ball")
    }


def list_signals(signals: Dict[str, VibrationSignal]) -> List[str]:
    """Return the names of the loaded signals."""
    return list(signals)
