"""Fault-size (severity) analysis for CWRU bearing recordings.

The CWRU dataset contains each fault type at three standard damage diameters
(0.007 in, 0.014 in, 0.021 in). Grouping the feature table by damage size shows
the classic damage-progression trend: vibration severity grows with defect size.

Functions:
* ``parse_fault_size`` — recover the damage diameter from a filename.
* ``severity_table`` — recordings with a known size and fault, with indicators.
* ``severity_trends`` — slope / correlation of an indicator versus size.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

#: Standard CWRU fault codes mapped to damage diameter (inches).
SIZE_CODES = {7: 0.007, 14: 0.014, 21: 0.021}

#: Indicators tracked for damage progression.
SEVERITY_FEATURES: List[str] = ["rms", "std", "peak", "kurtosis", "crest_factor"]


def parse_fault_size(name) -> Optional[float]:
    """Extract the fault diameter (inches) from a filename or recording name.

    Recognises ``IR007``/``OR014``/``B021`` style codes as well as explicit
    decimals such as ``0.007``. Returns ``None`` when no size is found.
    """
    text = Path(str(name)).name.lower()
    for size in (0.007, 0.014, 0.021):
        if f"{size:.3f}" in text:
            return size
    match = re.search(r"(?:ir|or|b)(\d{1,3})", text)
    if match:
        code = int(match.group(1))
        if code in SIZE_CODES:
            return SIZE_CODES[code]
    match = re.search(r"\b(\d{3})\b", text)
    if match:
        code = int(match.group(1))
        if code in SIZE_CODES:
            return SIZE_CODES[code]
    return None


def severity_table(table: pd.DataFrame, conditions: Dict[str, str],
                   sources: Optional[Dict[str, str]] = None,
                   features: Optional[Iterable[str]] = None) -> pd.DataFrame:
    """Keep only faulty recordings with a known damage size, with indicators.

    ``sources`` optionally maps recording name -> original filename, used as a
    fallback when the size cannot be read from the recording name itself.
    """
    features = list(features) if features is not None else list(SEVERITY_FEATURES)
    sources = sources or {}
    rows = []
    for name in table.index:
        condition = conditions.get(name, "unknown")
        if condition == "healthy":
            continue
        size = parse_fault_size(name) or parse_fault_size(sources.get(name, ""))
        if size is None:
            continue
        row = table.loc[name]
        entry = {"recording": str(name), "condition": condition, "size_in": float(size)}
        for feature in features:
            value = row.get(feature, np.nan)
            entry[feature] = float(value) if pd.notna(value) else np.nan
        rows.append(entry)
    if not rows:
        return pd.DataFrame(columns=["recording", "condition", "size_in"] + features)
    return (pd.DataFrame(rows)
            .sort_values(["condition", "size_in"])
            .reset_index(drop=True))


def severity_trends(severity: pd.DataFrame, features: Optional[Iterable[str]] = None) -> pd.DataFrame:
    """Linear trend (slope + Pearson r) of each indicator vs. damage size.

    A positive slope and strong positive correlation mean the indicator tracks
    fault severity, i.e. it grows as the defect gets larger.
    """
    if severity.empty:
        return pd.DataFrame(columns=["condition", "feature", "slope", "correlation",
                                     "increasing", "levels"])
    features = [f for f in (list(features) if features is not None else list(SEVERITY_FEATURES))
                if f in severity.columns]
    rows = []
    for condition, group in severity.groupby("condition"):
        sizes = group["size_in"].to_numpy(dtype=float)
        if len(group) < 2 or np.ptp(sizes) == 0:
            continue
        for feature in features:
            values = group[feature].to_numpy(dtype=float)
            if np.isnan(values).all():
                continue
            valid = ~np.isnan(values)
            if valid.sum() < 2:
                continue
            slope = float(np.polyfit(sizes[valid], values[valid], 1)[0])
            correlation = float(np.corrcoef(sizes[valid], values[valid])[0, 1])
            rows.append({
                "condition": condition,
                "feature": feature,
                "slope": slope,
                "correlation": correlation,
                "increasing": bool(slope > 0),
                "levels": int(valid.sum()),
            })
    if not rows:
        return pd.DataFrame(columns=["condition", "feature", "slope", "correlation",
                                     "increasing", "levels"])
    return pd.DataFrame(rows).sort_values(["condition", "feature"]).reset_index(drop=True)


def summarise_severity(severity: pd.DataFrame, feature: str = "rms") -> List[str]:
    """Human-readable lines describing the damage-progression trend."""
    if severity.empty:
        return ["No damage-size information found (severity trend not available for this dataset)."]

    trends = severity_trends(severity, [feature])
    lines: List[str] = []
    for _, row in trends.iterrows():
        direction = "increases" if row["increasing"] else "decreases"
        strength = ("strongly" if abs(row["correlation"]) >= 0.8 else
                    "moderately" if abs(row["correlation"]) >= 0.5 else "weakly")
        lines.append(
            f"{row['condition'].replace('_', ' ').title()}: {feature} {direction} "
            f"with damage size ({strength} correlated, r = {row['correlation']:+.2f})"
        )
    return lines
