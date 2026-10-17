"""Loading utilities for bearing vibration signals.

Supports CSV and MATLAB ``.mat`` recordings as well as a synthetic signal
generator so the pipeline can run without the full dataset.
"""

from __future__ import annotations

import io
import re
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


def _select_mat_key(keys: List[str], hints: Optional[List[str]] = None) -> Optional[str]:
    """Pick the vibration channel from a CWRU ``.mat`` file.

    CWRU recordings expose ``*_DE_time`` (drive end), ``*_FE_time`` (fan end),
    ``*_BA_time`` (base) and ``*RPM`` arrays. Preference order: drive end,
    fan end, base acceleration, then any other candidate.

    ``hints`` are numeric tokens from the filename (e.g. ``["175", "1", "014"]``
    for ``IR014_1_175.mat``). When a hint matches a channel name the correct
    channel is selected even if the file also contains data from another file.
    """
    suffixes = ("_DE_time", "_FE_time", "_BA_time")
    for hint in hints or []:
        for suffix in suffixes:
            for k in keys:
                if k.endswith(suffix) and hint in k:
                    return k
    for suffix in suffixes:
        for k in keys:
            if k.endswith(suffix):
                return k
    return keys[0] if keys else None


def infer_condition(filename: str) -> str:
    """Infer a bearing condition label from a CWRU-style filename.

    Recognises ``IR`` (inner race), ``OR`` (outer race), ``B`` (ball) and
    ``normal``/``healthy`` (healthy baseline). Codes must be followed by a
    digit (e.g. ``IR007``) or appear as keywords to avoid false matches.
    Returns ``"unknown"`` otherwise.
    """
    key = Path(filename).name.lower()
    if "normal" in key or "healthy" in key or "baseline" in key:
        return "healthy"
    if "inner" in key or re.search(r"(^|[^a-z])ir\d", key):
        return "inner_race"
    if "outer" in key or re.search(r"(^|[^a-z])or\d", key):
        return "outer_race"
    if "ball" in key or re.search(r"(^|[^a-z])b\d", key):
        return "ball"
    return "unknown"


def _signal_from_matdict(data: dict, path_stem: str, fs: float,
                         condition: str, key: Optional[str] = None,
                         name: Optional[str] = None) -> VibrationSignal:
    """Build a ``VibrationSignal`` from a loaded ``.mat`` dictionary."""
    candidates = [
        k for k, v in data.items()
        if not k.startswith("__") and isinstance(v, np.ndarray) and np.issubdtype(v.dtype, np.floating)
    ]
    if not candidates:
        raise ValueError(f"No numeric array found in {path_stem}")
    hints = list(reversed(re.findall(r"\d+", path_stem)))
    selected = key or _select_mat_key(candidates, hints)
    return VibrationSignal(
        name=name or f"{path_stem}_{selected}",
        condition=condition,
        signal=data[selected].ravel(),
        fs=fs,
    )


def load_mat(path: Union[str, Path], key: Optional[str] = None, fs: float = 12000.0,
             name: Optional[str] = None, condition: str = "unknown") -> VibrationSignal:
    """Load a MATLAB ``.mat`` recording.

    Uses ``key`` when given, otherwise selects the CWRU drive-end channel
    (``*_DE_time``) that best matches the filename, else the first
    floating-point array.
    """
    if loadmat is None:  # pragma: no cover
        raise ImportError("scipy is required to read .mat files")
    path = Path(path)
    return _signal_from_matdict(
        loadmat(path), path.stem, fs, condition, key=key, name=name
    )


def load_mat_bytes(content: bytes, filename: str, fs: float = 12000.0,
                   condition: Optional[str] = None,
                   name: Optional[str] = None) -> VibrationSignal:
    """Load a MATLAB ``.mat`` recording from raw bytes (e.g. an upload)."""
    if loadmat is None:  # pragma: no cover
        raise ImportError("scipy is required to read .mat files")
    data = loadmat(io.BytesIO(content))
    stem = Path(filename).stem
    return _signal_from_matdict(
        data, stem, fs, condition or infer_condition(filename), name=name
    )


def load_csv_bytes(content: bytes, filename: str, fs: float = 12000.0,
                   column: str = "amplitude", condition: Optional[str] = None,
                   name: Optional[str] = None) -> VibrationSignal:
    """Load a CSV recording from raw bytes.

    Uses ``column`` when present, otherwise the first column. Falls back to a
    whitespace-delimited read for headerless files.
    """
    stem = Path(filename).stem
    try:
        frame = pd.read_csv(io.BytesIO(content))
    except Exception:  # noqa: BLE001 - fall back to whitespace parsing
        frame = pd.read_csv(io.BytesIO(content), sep=r"\s+", header=None)
    numeric = frame.select_dtypes("number")
    if column in numeric.columns:
        series = numeric[column]
    elif len(numeric.columns):
        series = numeric.iloc[:, 0]
    else:
        raise ValueError(f"No numeric column found in {filename}")
    return VibrationSignal(
        name=name or stem,
        condition=condition or infer_condition(filename),
        signal=series.to_numpy(),
        fs=fs,
    )


def load_npz_bytes(content: bytes, filename: str, fs: float = 12000.0,
                   condition: Optional[str] = None,
                   name: Optional[str] = None) -> VibrationSignal:
    """Load a NumPy ``.npz`` archive from raw bytes (first array / ``*_DE_time``)."""
    stem = Path(filename).stem
    with np.load(io.BytesIO(content)) as archive:
        keys = [k for k in archive.files]
        if not keys:
            raise ValueError(f"No arrays found in {filename}")
        hints = list(reversed(re.findall(r"\d+", stem)))
        selected = _select_mat_key(keys, hints) or keys[0]
        signal = np.asarray(archive[selected])
    return VibrationSignal(
        name=name or f"{stem}_{selected}",
        condition=condition or infer_condition(filename),
        signal=signal.ravel(),
        fs=fs,
    )


def load_uploaded(content: bytes, filename: str, fs: float = 12000.0,
                  condition: Optional[str] = None) -> VibrationSignal:
    """Load an uploaded recording, dispatching on the file extension."""
    extension = Path(filename).suffix.lower()
    if extension == ".mat":
        return load_mat_bytes(content, filename, fs=fs, condition=condition)
    if extension == ".csv":
        return load_csv_bytes(content, filename, fs=fs, condition=condition)
    if extension == ".npz":
        return load_npz_bytes(content, filename, fs=fs, condition=condition)
    if extension in {".txt", ".dat"}:
        return load_csv_bytes(content, filename, fs=fs, condition=condition)
    raise ValueError(f"Unsupported file type: {extension}")


def load_directory(root: Union[str, Path], fs: float = 12000.0,
                   condition: Optional[str] = None) -> Dict[str, VibrationSignal]:
    """Load every CSV/``.mat``/``.npz`` file under ``root``.

    The condition defaults to the immediate parent folder name; if that name is
    not a recognised condition, it is inferred from the filename instead.
    """
    root = Path(root)
    signals: Dict[str, VibrationSignal] = {}
    for path in sorted(root.rglob("*")):
        if path.suffix.lower() not in {".csv", ".mat", ".npz"}:
            continue
        if condition is not None:
            cond = condition
        else:
            inferred = infer_condition(path.name)
            cond = inferred if inferred != "unknown" else path.parent.name
        try:
            sig = _load_path(path, fs=fs, condition=cond)
        except Exception as exc:  # noqa: BLE001 - keep loading the rest
            print(f"[data_loading] skipped {path}: {exc}")
            continue
        signals[sig.name] = sig
    return signals


def _load_path(path: Path, fs: float, condition: str) -> VibrationSignal:
    """Load a single recording from disk, dispatching on its extension."""
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return load_csv(path, fs=fs, condition=condition)
    if suffix == ".mat":
        return load_mat(path, fs=fs, condition=condition)
    if suffix == ".npz":
        return load_npz_bytes(path.read_bytes(), path.name, fs=fs, condition=condition)
    raise ValueError(f"Unsupported file type: {suffix}")


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


def condition_map(signals: Dict[str, object]) -> Dict[str, str]:
    """Map recording name -> condition label.

    Uses the ``condition`` attribute of loaded signals and falls back to
    inferring the condition from the recording name.
    """
    mapping: Dict[str, str] = {}
    for name, obj in signals.items():
        label = getattr(obj, "condition", None)
        if not label or label == "unknown":
            label = infer_condition(str(name))
        mapping[name] = label
    return mapping
