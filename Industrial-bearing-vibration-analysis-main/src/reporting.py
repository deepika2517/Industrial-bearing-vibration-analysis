"""Generate a vibration-analysis report (HTML + Markdown).

The report collects the time-domain and frequency-domain findings, embeds the
generated figures, and adds a short interpretation for machine-condition
monitoring.
"""

from __future__ import annotations

import base64
import io
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from . import comparison, severity, statistical_tests


def _fig_to_base64(fig) -> str:
    """Render a matplotlib figure to an inline base64 PNG string."""
    import matplotlib.pyplot as plt

    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _img_tag(fig) -> str:
    return f'<img alt="figure" src="data:image/png;base64,{_fig_to_base64(fig)}"/>'


def interpret(table: pd.DataFrame, stats: Optional[Dict[str, object]] = None,
              severity_frame: Optional[pd.DataFrame] = None) -> List[str]:
    """Rule-based interpretation of the comparison table.

    When ``stats`` (from :func:`statistical_tests.summarize`) and/or
    ``severity_frame`` (from :func:`severity.severity_table`) are supplied the
    interpretation is extended with significance and damage-progression
    statements.
    """
    lines: List[str] = []
    index = list(table.index)
    healthy = None
    for candidate in index:
        key = str(candidate).lower()
        if "healthy" in key or "normal" in key:
            healthy = candidate
            break

    if healthy is not None:
        base = table.loc[healthy]
        lines.append(
            "Healthy baseline — RMS {rms:.3f}, kurtosis {k:.3f}, "
            "crest factor {cf:.3f}.".format(
                rms=base["rms"], k=base["kurtosis"], cf=base["crest_factor"]
            )
        )

    by_rms = table["rms"].sort_values(ascending=False)
    by_kurt = table["kurtosis"].sort_values(ascending=False)
    by_cf = table["crest_factor"].sort_values(ascending=False)
    lines.append(
        f"Highest vibration energy (RMS): {by_rms.index[0]} "
        f"({by_rms.iloc[0]:.3f})."
    )
    lines.append(
        f"Most impulsive signature (kurtosis): {by_kurt.index[0]} "
        f"({by_kurt.iloc[0]:.3f})."
    )
    lines.append(
        f"Highest crest factor (peak/RMS): {by_cf.index[0]} "
        f"({by_cf.iloc[0]:.3f})."
    )

    for condition in index:
        if condition == healthy:
            continue
        row = table.loc[condition]
        notes: List[str] = []
        if healthy is not None:
            base_rms = table.loc[healthy, "rms"]
            if base_rms:
                delta = (row["rms"] / base_rms - 1.0) * 100.0
                direction = "higher" if delta >= 0 else "lower"
                notes.append(f"RMS {direction} by {abs(delta):.0f}% vs healthy")
        dom = row.get("dominant_freq_1", np.nan)
        if pd.notna(dom):
            notes.append(f"dominant frequency ~= {dom:.0f} Hz")
        notes.append(f"kurtosis {row['kurtosis']:.2f}")
        lines.append(f"{condition.replace('_', ' ').title()}: " + "; ".join(notes) + ".")

    if stats:
        omnibus = stats.get("omnibus")
        if omnibus is not None and not omnibus.empty:
            anova = omnibus[(omnibus["test"] == "One-way ANOVA") & omnibus["significant"]]
            significant = anova["feature"].drop_duplicates().tolist()
            if significant:
                lines.append(
                    "Statistically significant across conditions (one-way ANOVA, "
                    "p < 0.05): " + ", ".join(significant) + "."
                )
            else:
                lines.append(
                    "No indicator reached significance across conditions "
                    "(one-way ANOVA, p >= 0.05)."
                )

        pairwise = stats.get("pairwise")
        if pairwise is not None and not pairwise.empty:
            notable = pairwise[pairwise["significant"] &
                               pairwise["effect"].isin(["large", "medium"])]
            for _, row in notable.head(3).iterrows():
                lines.append(
                    f"{row['feature']}: {row['reference']} vs {row['compared_to']} "
                    f"Cohen's d = {row['cohens_d']:+.2f} ({row['effect']} effect, "
                    f"p = {row['p_value']:.1e})."
                )

    if severity_frame is not None and not severity_frame.empty:
        lines.extend(severity.summarise_severity(severity_frame, "rms"))

    lines.append(
        "Faulty bearings show elevated RMS, kurtosis and crest factor relative "
        "to the healthy baseline, with extra spectral energy at defect-related "
        "frequencies — the expected signature of inner-race, outer-race and "
        "ball faults. These indicators support condition-based monitoring and "
        "early fault detection."
    )
    return lines


def _figure_rows(output: Path, signals, fs: float):
    """Generate and save every report figure; return (filename, caption) list."""
    out: List[Dict[str, str]] = []

    specs = [
        ("waveforms.png", "Raw vibration waveforms by condition",
         lambda p: comparison.plot_waveforms(signals, fs, save_path=p)),
        ("spectra_comparison.png", "Frequency-domain comparison (FFT amplitude spectra)",
         lambda p: comparison.plot_spectra(signals, fs, save_path=p)),
        ("rms_comparison.png", "RMS comparison across conditions",
         lambda p: comparison.plot_comparison(signals, fs, metric="rms", save_path=p)),
        ("kurtosis_comparison.png", "Kurtosis comparison across conditions",
         lambda p: comparison.plot_comparison(signals, fs, metric="kurtosis", save_path=p)),
        ("crest_factor_comparison.png", "Crest-factor comparison across conditions",
         lambda p: comparison.plot_comparison(signals, fs, metric="crest_factor", save_path=p)),
    ]
    for filename, caption, make in specs:
        path = output / filename
        fig, _ = make(path)
        out.append({"filename": filename, "caption": caption, "html": _img_tag(fig)})
    return out


def generate_report(signals, fs: float, output_dir: str | Path,
                    title: str = "Industrial Bearing Vibration Analysis Report",
                    conditions: Optional[Dict[str, str]] = None) -> Dict[str, Path]:
    """Build ``report.html`` and ``report.md`` with figures and findings.

    When ``conditions`` (recording name -> condition) is provided the report is
    extended with statistical significance tests and a fault-severity trend.
    """
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    table = comparison.build_comparison_table(signals, fs).round(4)
    condition_table = comparison.aggregate_by_condition(table, conditions) if conditions else table
    table_path = comparison.save_comparison(table, output / "comparison_table.csv")
    if conditions:
        comparison.save_comparison(condition_table, output / "condition_summary.csv")
    figures = _figure_rows(output, signals, fs)

    stats: Optional[Dict[str, object]] = None
    severity_frame = pd.DataFrame()
    if conditions:
        arrays = comparison._as_signal_dict(signals)
        try:
            stats = statistical_tests.summarize(
                statistical_tests.window_features(arrays, conditions, fs)
            )
        except Exception:  # noqa: BLE001 - statistics are optional
            stats = None
        try:
            severity_frame = severity.severity_table(table, conditions)
        except Exception:  # noqa: BLE001 - severity is optional
            severity_frame = pd.DataFrame()

    lines = interpret(condition_table, stats=stats, severity_frame=severity_frame)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

    # --- shared section content ------------------------------------------ #
    interp_html = "\n".join(f"<li>{line}</li>" for line in lines)
    interp_md = "\n".join(f"- {line}" for line in lines)
    md_figs = "\n\n".join(
        f"![{f['caption']}]({f['filename']})\n\n*{f['caption']}*" for f in figures
    )
    feature_md = _to_markdown_table(condition_table)
    detail_html = detail_md = ""
    if conditions:
        detail_html = ("\n<p>Recording-level features (before averaging):</p>\n"
                       + table.to_html(border=1))
        detail_md = ("\n\nRecording-level features (before averaging):\n\n"
                     + _to_markdown_table(table))

    stats_html = stats_md = ""
    if stats and stats.get("omnibus") is not None and not stats["omnibus"].empty:
        omni = _fmt_p(stats["omnibus"])
        head = f"{stats['n_windows']} analysis windows across " \
               f"{len(stats['conditions'])} conditions."
        stats_html += f"<p>{head}</p>\n{omni.to_html(border=1, index=False)}"
        stats_md += f"{head}\n\n{_md_table(omni)}"
        pair = stats.get("pairwise")
        if pair is not None and not pair.empty:
            reference = str(pair["reference"].iloc[0])
            pair_disp = _fmt_p(pair)
            stats_html += (
                f"\n<p>Pairwise Welch t-tests against <b>{reference}</b> "
                f"(Cohen's d = effect size):</p>\n"
                f"{pair_disp.to_html(border=1, index=False)}"
            )
            stats_md += (
                f"\n\nPairwise Welch t-tests against **{reference}** "
                f"(Cohen's d = effect size):\n\n{_md_table(pair_disp)}"
            )

    severity_html = severity_md = ""
    if not severity_frame.empty:
        trends = severity.severity_trends(severity_frame, ["rms", "kurtosis", "crest_factor"])
        severity_html += (
            f"<p>Damage size recovered from the CWRU fault codes "
            f"(0.007 / 0.014 / 0.021 in):</p>\n"
            f"{severity_frame.to_html(border=1, index=False)}\n"
            f"<p>Linear trend of each indicator against damage size "
            f"(positive slope = grows with severity):</p>\n"
            f"{trends.to_html(border=1, index=False)}"
        )
        severity_md += (
            f"Damage size recovered from the CWRU fault codes "
            f"(0.007 / 0.014 / 0.021 in):\n\n{_md_table(severity_frame)}\n\n"
            f"Linear trend of each indicator against damage size:\n\n{_md_table(trends)}"
        )


    # --- assemble report sections ---------------------------------------- #
    condition_list = ", ".join(sorted(set(conditions.values()))) if conditions else "not specified"
    objective = ("<p>Analyse vibration signals from rotating machinery to distinguish "
                 "normal and faulty bearing operating conditions using statistical and "
                 "frequency-domain analysis (CWRU bearing dataset).</p>")
    methodology = ("<p>Each recording is detrended, band-pass filtered and optionally "
                   "normalised, then summarised with time-domain indicators (mean, RMS, "
                   "variance, standard deviation, peak, kurtosis, crest factor, shape and "
                   "impulse factors) and frequency-domain features (NumPy FFT magnitude "
                   "spectrum, dominant frequencies, spectral centroid and band-energy "
                   "ratios).</p>")
    conclusion = ("<p>Statistical indicators combined with FFT-based frequency analysis "
                  "clearly separate healthy bearings from inner-race, outer-race and ball "
                  "faults. RMS, kurtosis and crest factor provide simple, interpretable "
                  "condition indicators, while the dominant frequencies and band-energy "
                  "ratios locate the defect signature in the spectrum — enabling "
                  "condition-based maintenance decisions.</p>")

    html_sections = [
        ("Objective", objective),
        ("Methodology", methodology),
        ("Time-domain characteristics", _section_figures(
            figures, ["waveforms.png", "rms_comparison.png",
                      "kurtosis_comparison.png", "crest_factor_comparison.png"])),
        ("Frequency-domain comparison", _section_figures(figures, ["spectra_comparison.png"])),
        ("Feature comparison table", condition_table.to_html(border=1) + detail_html),
    ]
    if stats_html:
        html_sections.append(("Statistical significance", stats_html))
    if severity_html:
        html_sections.append(("Fault-severity (damage-progression) trend", severity_html))
    html_sections.append(("Interpretation for machine-condition monitoring",
                          f"<ul>\n{interp_html}\n</ul>"))
    html_sections.append(("Conclusion", conclusion))

    body_html = "\n".join(
        f"<h2>{number}. {name}</h2>\n{content}"
        for number, (name, content) in enumerate(html_sections, 1)
    )

    # --- HTML report (self-contained) ---
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>{title}</title>
<style>
 body {{ font-family: Segoe UI, Arial, sans-serif; margin: 2rem auto; max-width: 900px;
        color: #1c1c1c; line-height: 1.5; }}
 h1, h2 {{ color: #16324f; }}
 h2 {{ border-bottom: 2px solid #e6ecf2; padding-bottom: 4px; margin-top: 2rem; }}
 table {{ border-collapse: collapse; margin: 1rem 0; font-size: 0.85rem; }}
 th, td {{ border: 1px solid #cfd8dc; padding: 4px 8px; text-align: right; }}
 th:first-child, td:first-child {{ text-align: left; }}
 th {{ background: #f4f7fa; }}
 figure {{ margin: 1.2rem 0; }}
 img {{ max-width: 100%; border: 1px solid #e0e0e0; border-radius: 6px; }}
 figcaption {{ font-size: 0.85rem; color: #555; font-style: italic; margin-top: 4px; }}
 code {{ background: #f2f4f6; padding: 1px 4px; border-radius: 3px; }}
 .meta {{ color: #555; font-size: 0.9rem; }}
</style>
</head>
<body>
<h1>{title}</h1>
<p class="meta">Generated: {timestamp} &nbsp;|&nbsp; Sampling frequency: {fs:.0f} Hz
 &nbsp;|&nbsp; Conditions: {condition_list} &nbsp;|&nbsp; Recordings: {len(signals)}</p>

{body_html}
</body>
</html>
"""
    html_path = output / "report.html"
    html_path.write_text(html, encoding="utf-8")

    # --- Markdown report ---
    md_sections = [
        ("Objective", objective.replace("<p>", "").replace("</p>", "")),
        ("Methodology", methodology.replace("<p>", "").replace("</p>", "")),
        ("Figures", md_figs),
        ("Feature comparison table", feature_md + detail_md),
    ]
    if stats_md:
        md_sections.append(("Statistical significance", stats_md))
    if severity_md:
        md_sections.append(("Fault-severity (damage-progression) trend", severity_md))
    md_sections.append(("Interpretation for machine-condition monitoring", interp_md))
    md_sections.append(("Conclusion", conclusion.replace("<p>", "").replace("</p>", "")))

    md_body = "\n\n".join(
        f"## {number}. {name}\n\n{content}"
        for number, (name, content) in enumerate(md_sections, 1)
    )
    md = f"""# {title}

*Generated: {timestamp} — sampling frequency {fs:.0f} Hz — conditions: {condition_list} — recordings: {len(signals)}*

{md_body}
"""
    md_path = output / "report.md"
    md_path.write_text(md, encoding="utf-8")

    return {"html": html_path, "markdown": md_path, "table": table_path}


def _section_figures(figures, names) -> str:
    """Concatenate selected embedded figures by filename."""
    selected = {f["filename"]: f for f in figures}
    blocks = []
    for name in names:
        f = selected.get(name)
        if f:
            blocks.append(f'<figure>{f["html"]}<figcaption>{f["caption"]}</figcaption></figure>')
    return "\n".join(blocks)


def _to_markdown_table(table: pd.DataFrame, index_label: str = "condition") -> str:
    """Render a DataFrame as a GitHub-style markdown table (no extra deps)."""
    display = table.reset_index().rename(columns={"index": index_label})
    return _md_table(display)


def _md_table(df: pd.DataFrame) -> str:
    """Render a DataFrame as a markdown table using its columns only."""
    columns = [str(c) for c in df.columns]
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = [header, separator]
    for _, values in df.iterrows():
        rows.append("| " + " | ".join(_fmt_cell(v) for v in values) + " |")
    return "\n".join(rows)


def _fmt_p(df: pd.DataFrame, column: str = "p_value") -> pd.DataFrame:
    """Return a copy of ``df`` with p-values rendered in scientific notation."""
    display = df.copy()
    if column in display.columns:
        display[column] = display[column].map(lambda value: f"{float(value):.2e}")
    return display


#: Public alias used by the dashboard.
format_pvalues = _fmt_p


def _fmt_cell(value) -> str:
    if isinstance(value, float):
        if value != value:  # NaN
            return "-"
        if value != 0 and abs(value) < 1e-3:
            return f"{value:.1e}"
        return f"{value:.4g}"
    return str(value)
