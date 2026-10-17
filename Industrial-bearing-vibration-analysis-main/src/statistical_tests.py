"""Statistical significance testing across bearing conditions.

Turns the "feature differs between conditions?" question into a testable
hypothesis:

* **Omnibus** — one-way ANOVA and the non-parametric Kruskal-Wallis test
  decide whether at least one condition differs from the others.
* **Pairwise** — Welch t-tests with Cohen's d compare every fault condition
  against the healthy baseline.

Signals are first cut into overlapping windows so each condition contributes
many observations, giving the tests adequate statistical power.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

from . import preprocessing, statistical_analysis

DEFAULT_FEATURES: List[str] = [
    "mean", "std", "variance", "rms", "peak", "peak_to_peak",
    "skewness", "kurtosis", "crest_factor", "shape_factor", "impulse_factor",
]


def window_features(signals: Dict[str, np.ndarray], conditions: Dict[str, str],
                    fs: float, window_sec: float = 0.25,
                    overlap: float = 0.5) -> pd.DataFrame:
    """Compute time-domain features for every window of every recording.

    Returns a DataFrame with one row per window plus ``recording`` and
    ``condition`` columns for grouping.
    """
    window = max(int(window_sec * fs), 64)
    rows = []
    for name, signal in signals.items():
        for segment in preprocessing.segment_signal(np.asarray(signal, float), window, overlap):
            if len(segment) < window:
                continue
            features = statistical_analysis.time_domain_features(segment)
            features["recording"] = name
            features["condition"] = conditions.get(name, "unknown")
            rows.append(features)
    return pd.DataFrame(rows)


def _groups(frame: pd.DataFrame, feature: str) -> List[np.ndarray]:
    """Valid (n >= 2) per-condition value arrays for ``feature``."""
    grouped = frame.groupby("condition")[feature]
    return [values.dropna().to_numpy() for _, values in grouped if values.notna().sum() >= 2]


def omnibus_tests(frame: pd.DataFrame, features: Optional[Iterable[str]] = None,
                  alpha: float = 0.05) -> pd.DataFrame:
    """One-way ANOVA + Kruskal-Wallis per feature across conditions."""
    from scipy.stats import f_oneway, kruskal

    features = list(features) if features is not None else list(DEFAULT_FEATURES)
    rows = []
    for feature in features:
        groups = _groups(frame, feature)
        if len(groups) < 2:
            continue
        for test_name, test in (("One-way ANOVA", f_oneway), ("Kruskal-Wallis", kruskal)):
            try:
                statistic, p_value = test(*groups)
            except Exception:  # noqa: BLE001 - degenerate input
                continue
            rows.append({
                "feature": feature,
                "test": test_name,
                "statistic": float(statistic),
                "p_value": float(p_value),
                "significant": bool(p_value < alpha),
            })
    if not rows:
        return pd.DataFrame(columns=["feature", "test", "statistic", "p_value", "significant"])
    return (pd.DataFrame(rows)
            .sort_values(["feature", "test"])
            .reset_index(drop=True))


def pairwise_vs_reference(frame: pd.DataFrame, features: Optional[Iterable[str]] = None,
                          reference: str = "healthy", alpha: float = 0.05) -> pd.DataFrame:
    """Welch t-test + Cohen's d: each condition against a reference group."""
    from scipy.stats import ttest_ind

    features = list(features) if features is not None else list(DEFAULT_FEATURES)
    conditions = list(dict.fromkeys(frame["condition"]))
    if reference not in conditions:
        reference = "healthy" if "healthy" in conditions else conditions[0]

    base = frame[frame["condition"] == reference]
    rows = []
    for condition in conditions:
        if condition == reference:
            continue
        other = frame[frame["condition"] == condition]
        for feature in features:
            a = base[feature].dropna().to_numpy()
            b = other[feature].dropna().to_numpy()
            if len(a) < 2 or len(b) < 2:
                continue
            try:
                t_stat, p_value = ttest_ind(a, b, equal_var=False)
            except Exception:  # noqa: BLE001 - degenerate input
                continue
            pooled = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1))
                             / (len(a) + len(b) - 2))
            cohen = float((b.mean() - a.mean()) / pooled) if pooled else 0.0
            rows.append({
                "feature": feature,
                "reference": reference,
                "compared_to": condition,
                "t_statistic": float(t_stat),
                "p_value": float(p_value),
                "cohens_d": cohen,
                "effect": _effect_label(abs(cohen)),
                "significant": bool(p_value < alpha),
            })
    if not rows:
        return pd.DataFrame(columns=["feature", "reference", "compared_to",
                                     "t_statistic", "p_value", "cohens_d",
                                     "effect", "significant"])
    return (pd.DataFrame(rows)
            .sort_values(["compared_to", "p_value"])
            .reset_index(drop=True))


def _effect_label(value: float) -> str:
    """Standard Cohen's d thresholds."""
    if value >= 0.8:
        return "large"
    if value >= 0.5:
        return "medium"
    if value >= 0.2:
        return "small"
    return "negligible"


def significant_features(table: pd.DataFrame, alpha: float = 0.05) -> List[str]:
    """Features whose p-value < alpha (sorted by significance)."""
    if table.empty or "p_value" not in table.columns:
        return []
    return (table[table["p_value"] < alpha]
            .sort_values("p_value")["feature"]
            .drop_duplicates()
            .tolist())


def summarize(frame: pd.DataFrame, alpha: float = 0.05) -> Dict[str, object]:
    """Run both test families and return a compact dict of results."""
    omnibus = omnibus_tests(frame, alpha=alpha)
    pairwise = pairwise_vs_reference(frame, alpha=alpha)
    return {
        "n_windows": int(len(frame)),
        "conditions": sorted(frame["condition"].unique().tolist()),
        "omnibus": omnibus,
        "pairwise": pairwise,
        "significant": significant_features(omnibus, alpha=alpha),
    }
