"""Streamlit dashboard for industrial bearing vibration analysis.

Self-contained: import a dataset (upload files, point to a folder, or use the
built-in demo), run the time-domain and frequency-domain analysis in the
browser, and download the results — no terminal commands required.
"""

import io
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import (  # noqa: E402
    comparison,
    data_loading,
    fft_analysis,
    preprocessing,
    reporting,
    severity,
    statistical_analysis,
    statistical_tests,
)

RESULTS_DIR = ROOT / "results"
CONDITIONS = ["healthy", "inner_race", "outer_race", "ball", "unknown"]

#: Consistent colour per bearing condition, reused across every chart.
CONDITION_COLORS = {
    "healthy": "#2e7d32",
    "inner_race": "#1565c0",
    "outer_race": "#ef6c00",
    "ball": "#c62828",
    "unknown": "#6d6d6d",
}


def condition_color(condition: str) -> str:
    """Colour for a condition label (falls back to grey)."""
    return CONDITION_COLORS.get(condition, "#6d6d6d")


# --------------------------------------------------------------------------- #
# Data acquisition
# --------------------------------------------------------------------------- #
def parse_uploads(uploaded, fs):
    """Load uploaded files into ``{name: {signal, condition, source}}``."""
    data = {}
    errors = []
    for file in uploaded:
        try:
            sig = data_loading.load_uploaded(file.getvalue(), file.name, fs=fs)
        except Exception as exc:  # noqa: BLE001 - report and continue
            errors.append(f"{file.name}: {exc}")
            continue
        data[sig.name] = {"signal": sig.signal, "condition": sig.condition, "source": file.name}
    return data, errors


def load_folders(folders, fs):
    """Load and merge recordings from several folders."""
    data, errors = {}, []
    for folder in folders:
        if not Path(folder).exists():
            errors.append(f"{folder}: folder not found")
            continue
        for name, signal in data_loading.load_directory(folder, fs=fs).items():
            key, i = name, 1
            while key in data:
                key, i = f"{name}#{i}", i + 1
            data[key] = {
                "signal": signal.signal,
                "condition": signal.condition,
                "source": str(Path(folder) / name),
            }
    return data, errors


def pick_folder():
    """Open a native folder picker (best effort, local use only)."""
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        selected = filedialog.askdirectory(title="Select a dataset folder")
        root.destroy()
        return selected or None
    except Exception:  # noqa: BLE001 - picker is optional
        return None


def load_demo(fs):
    demo = data_loading.make_demo_dataset(fs=fs)
    return {
        name: {"signal": s.signal, "condition": s.condition, "source": "demo"}
        for name, s in demo.items()
    }


def analyze(data, fs, signature):
    """Preprocess + build the feature table, memoised in session state."""
    cache = st.session_state
    if cache.get("_analysis_sig") != (signature, fs) or "_table" not in cache:
        processed = {}
        for name, item in data.items():
            processed[name] = preprocessing.preprocess_pipeline(
                item["signal"], fs, normalize_method=None
            )
        cache["_processed"] = processed
        cache["_table"] = comparison.build_comparison_table(processed, fs)
        cache["_analysis_sig"] = (signature, fs)
    return cache["_processed"], cache["_table"]


def compute_stats(processed, conditions, fs, signature):
    """Statistical significance tests (memoised)."""
    cache = st.session_state
    key = (signature, tuple(sorted(conditions.items())), fs, "stats")
    if cache.get("_stats_key") != key:
        try:
            frame = statistical_tests.window_features(processed, conditions, fs)
            cache["_stats"] = statistical_tests.summarize(frame)
        except Exception as exc:  # noqa: BLE001 - tests are best effort
            cache["_stats"] = {"error": str(exc)}
        cache["_stats_key"] = key
    return cache["_stats"]


def compute_severity(table, conditions, signature):
    """Damage-size (severity) table (memoised)."""
    cache = st.session_state
    key = (signature, tuple(sorted(conditions.items())), "severity")
    if cache.get("_severity_key") != key:
        try:
            cache["_severity"] = severity.severity_table(table, conditions)
        except Exception:  # noqa: BLE001 - severity is best effort
            cache["_severity"] = pd.DataFrame()
        cache["_severity_key"] = key
    return cache["_severity"]


def build_results(processed, table, fs, signature, conditions):
    """Generate report + figures and return downloadable byte payloads."""
    rep = reporting.generate_report(
        processed, fs, RESULTS_DIR,
        conditions={name: conditions.get(name, "unknown") for name in processed},
    )
    figures = {p.name: p.read_bytes() for p in sorted(RESULTS_DIR.glob("*.png"))}
    csv_bytes = table.assign(condition=[conditions[n] for n in table.index]) \
        .to_csv(index=True).encode("utf-8")
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, blob in figures.items():
            zf.writestr(name, blob)
        zf.writestr("comparison_table.csv", csv_bytes)
        zf.writestr("report.html", Path(rep["html"]).read_bytes())
        zf.writestr("report.md", Path(rep["markdown"]).read_bytes())
    return {
        "sig": signature,
        "html": Path(rep["html"]).read_bytes(),
        "md": Path(rep["markdown"]).read_bytes(),
        "csv": csv_bytes,
        "zip": archive.getvalue(),
        "figures": figures,
    }


# --------------------------------------------------------------------------- #
# Plot helpers
# --------------------------------------------------------------------------- #
def plotly_waveform(signal, fs, max_samples, color="#1565c0"):
    import plotly.graph_objects as go

    count = min(int(max_samples), len(signal))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=np.arange(count) / fs, y=signal[:count], mode="lines",
                             name="amplitude", line=dict(color=color, width=1)))
    fig.update_layout(
        title="Raw vibration waveform", xaxis_title="Time (s)", yaxis_title="Amplitude",
        height=380, margin=dict(t=40, b=40), plot_bgcolor="white",
        paper_bgcolor="white", font=dict(color="#16324f"),
    )
    return fig


def plotly_spectrum(processed, names, fs, max_freq, conditions=None):
    import plotly.graph_objects as go

    conditions = conditions or {}
    fig = go.Figure()
    for name in names:
        freqs, mag = fft_analysis.compute_fft(processed[name], fs)
        mask = freqs <= max_freq
        fig.add_trace(go.Scatter(
            x=freqs[mask], y=mag[mask], mode="lines", name=name,
            line=dict(color=condition_color(conditions.get(name, "unknown")), width=1),
        ))
    fig.update_layout(
        title="FFT amplitude spectra", xaxis_title="Frequency (Hz)", yaxis_title="Amplitude",
        legend_title="Recording", height=420, margin=dict(t=40, b=40),
        plot_bgcolor="white", paper_bgcolor="white", font=dict(color="#16324f"),
    )
    return fig


def plotly_bars(table, metric, conditions=None):
    import plotly.graph_objects as go

    conditions = conditions or {}
    order = list(table.index)
    fig = go.Figure(go.Bar(
        x=order,
        y=[table.loc[name, metric] for name in order],
        marker_color=[condition_color(conditions.get(name, "unknown")) for name in order],
        text=[conditions.get(name, "unknown") for name in order],
    ))
    fig.update_layout(
        title=f"{metric} by recording (colour = condition)",
        xaxis_title="recording", yaxis_title=metric, height=420, showlegend=False,
        plot_bgcolor="white", paper_bgcolor="white", font=dict(color="#16324f"),
    )
    return fig


def plotly_pvalues(omnibus):
    import plotly.graph_objects as go

    anova = omnibus[omnibus["test"] == "One-way ANOVA"].copy()
    if anova.empty:
        return None
    anova["neg_log10_p"] = -np.log10(anova["p_value"].clip(lower=1e-300))
    anova = anova.sort_values("neg_log10_p")
    colors = ["#2e7d32" if sig else "#bdbdbd" for sig in anova["significant"]]
    fig = go.Figure(go.Bar(x=anova["neg_log10_p"], y=anova["feature"],
                           orientation="h", marker_color=colors))
    fig.add_vline(x=-np.log10(0.05), line_dash="dash", line_color="#c62828",
                  annotation_text="p = 0.05")
    fig.update_layout(
        title="One-way ANOVA: is the feature different across conditions?",
        xaxis_title="-log10(p)", yaxis_title="", height=420,
        margin=dict(t=40, b=40), plot_bgcolor="white", paper_bgcolor="white",
        font=dict(color="#16324f"),
    )
    return fig


def plotly_severity(severity_frame, metric, conditions=None):
    import plotly.express as px

    conditions = conditions or {}
    frame = severity_frame.copy()
    frame["condition"] = frame["condition"].map(lambda c: conditions.get(c, c))
    fig = px.bar(
        frame, x="size_in", y=metric, color="condition", barmode="group",
        color_discrete_map=CONDITION_COLORS,
        title=f"{metric} vs damage size (0.007 / 0.014 / 0.021 in)",
        labels={"size_in": "damage size (in)"},
    )
    fig.update_layout(height=420, plot_bgcolor="white", paper_bgcolor="white",
                      font=dict(color="#16324f"))
    return fig



# --------------------------------------------------------------------------- #
# Styling
# --------------------------------------------------------------------------- #
def inject_css():
    st.markdown(
        """
        <style>
        .block-container {padding-top: 1.4rem; padding-bottom: 2rem;}
        .hero {background: linear-gradient(120deg,#0f2027,#203a43,#2c5364);
               color:#ffffff; padding: 1.4rem 1.7rem; border-radius: 16px;
               margin-bottom: 1.2rem; box-shadow: 0 6px 18px rgba(15,32,39,.25);}
        .hero h1 {color:#ffffff; margin:0; font-size:1.7rem; letter-spacing:.2px;}
        .hero p {color:#cfe3ea; margin:.35rem 0 0; font-size:.95rem;}
        [data-testid="stMetricValue"] {color:#16324f;}
        [data-testid="stMetricLabel"] {color:#556677;}
        .stTabs [data-baseweb="tab-list"] {gap:6px;}
        .stTabs [data-baseweb="tab"] {background:#eef3f7; border-radius:8px 8px 0 0;
                                      padding:6px 14px;}
        .stTabs [aria-selected="true"] {background:#16324f !important; color:#fff !important;}
        h2, h3 {color:#16324f;}
        </style>
        """,
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------- #
# App
# --------------------------------------------------------------------------- #
def main():
    st.set_page_config(page_title="Bearing Vibration Analysis", page_icon="📊", layout="wide")
    inject_css()
    st.markdown(
        '<div class="hero"><h1>Industrial Bearing Vibration Analysis</h1>'
        '<p>CWRU drive-end data · time-domain indicators · FFT spectra · '
        'significance testing · damage-progression trend</p></div>',
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.header("Data source")
        source_mode = st.radio(
            "Where is the data?",
            ["Upload files", "Folder path(s)", "Built-in demo"],
            index=0,
        )
        fs = st.number_input("Sampling frequency (Hz)", min_value=100.0, value=48000.0, step=1000.0)
        if source_mode == "Folder path(s)":
            if st.button("📁 Browse for a folder…", width="stretch"):
                picked = pick_folder()
                if picked:
                    current = st.session_state.get("folder_paths", str(ROOT / "data" / "raw"))
                    st.session_state["folder_paths"] = (str(current).rstrip("\n") + "\n" + picked).strip()
                    st.rerun()
            folder_text = st.text_area(
                "Folder paths (one per line)",
                value=str(ROOT / "data" / "raw"),
                key="folder_paths",
                height=120,
                help="Add as many folders as you like — one path per line.",
            )
        st.divider()
        st.header("Display")
        max_samples = st.slider("Waveform samples", 500, 40000, 4000, step=500)
        max_freq = st.slider("Spectrum view limit (Hz)", 100, int(fs / 2), min(2000, int(fs / 2)), step=100)

    data, errors = {}, []
    uploaded = None

    if source_mode == "Upload files":
        uploaded = st.file_uploader(
            "Upload recordings (.mat, .csv, .npz)",
            type=["mat", "csv", "npz", "txt", "dat"],
            accept_multiple_files=True,
        )
        if not uploaded:
            st.info(
                "Upload one or more bearing recordings. The condition is inferred from "
                "the filename (IR→inner race, OR→outer race, B→ball, Normal→healthy) "
                "and you can override it below."
            )
            st.stop()
        signature = tuple(sorted((f.name, f.size) for f in uploaded)) + (fs,)
        cache = st.session_state
        if cache.get("_upload_sig") != signature:
            data, errors = parse_uploads(uploaded, fs)
            cache["_upload_data"] = data
            cache["_upload_sig"] = signature
            cache["_results"] = None
        else:
            data = cache["_upload_data"]
    elif source_mode == "Folder path(s)":
        folders = [line.strip() for line in folder_text.splitlines() if line.strip()]
        if not folders:
            st.error("Add at least one folder path.")
            st.stop()
        data, errors = load_folders(folders, fs)
        if not data:
            st.error("No CSV/.mat/.npz recordings found in those folders.")
            st.stop()
        signature = ("folders", tuple(folders), tuple(sorted(data)), fs)
    else:
        data = load_demo(fs)
        signature = ("demo", tuple(sorted(data)), fs)

    for message in errors:
        st.warning(f"Skipped {message}")

    # Optional condition overrides
    with st.expander("Assign / correct conditions (optional)", expanded=False):
        cols = st.columns(3)
        for i, name in enumerate(data):
            inferred = data[name]["condition"]
            index = CONDITIONS.index(inferred) if inferred in CONDITIONS else len(CONDITIONS) - 1
            choice = cols[i % 3].selectbox(name, CONDITIONS, index=index, key=f"cond_{name}")
            data[name]["condition"] = choice

    conditions = {name: item["condition"] for name, item in data.items()}

    with st.spinner("Analysing recordings..."):
        processed, table = analyze(data, fs, signature)

    row_conditions = [conditions.get(n, "unknown") for n in table.index]
    table_view = table.copy()
    table_view.insert(0, "condition", row_conditions)
    condition_table = comparison.aggregate_by_condition(table, conditions)

    with st.spinner("Running significance tests..."):
        stats = compute_stats(processed, conditions, fs, signature)
        severity_frame = compute_severity(table, conditions, signature)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Recordings", len(data))
    m2.metric("Conditions", len(set(conditions.values())))
    m3.metric("Sampling rate", f"{fs:,.0f} Hz")
    m4.metric("Source", source_mode)

    tab_wave, tab_spec, tab_feat, tab_stats, tab_sev, tab_report = st.tabs(
        ["Raw Waveform", "FFT Spectrum", "Feature Comparison",
         "Statistics", "Severity", "Results & Report"]
    )

    with tab_wave:
        sel = st.selectbox("Recording", list(data), index=0)
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(
                plotly_waveform(data[sel]["signal"], fs, max_samples,
                                color=condition_color(data[sel]["condition"])),
                width="stretch",
            )
        with c2:
            st.subheader("Time-domain features")
            feats = statistical_analysis.time_domain_features(processed[sel])
            st.dataframe(pd.DataFrame({"value": pd.Series(feats)}), width="stretch")
        st.caption(f"Condition: {data[sel]['condition']} — {len(data[sel]['signal']):,} samples")

    with tab_spec:
        selected = st.multiselect("Overlay recordings", list(data), default=list(data)[:4])
        if selected:
            st.plotly_chart(
                plotly_spectrum(processed, selected, fs, max_freq, conditions),
                width="stretch",
            )
            st.subheader("Dominant frequencies (Hz)")
            dom = {
                name: list(np.round(fft_analysis.dominant_frequencies(processed[name], fs, top_n=5), 2))
                for name in selected
            }
            st.dataframe(pd.DataFrame(dom), width="stretch")
        else:
            st.info("Select at least one recording.")

    with tab_feat:
        st.subheader("Condition summary (mean of recordings)")
        st.dataframe(condition_table.round(4), width="stretch")
        st.subheader("Comparison table (per recording)")
        st.dataframe(table_view.round(4), width="stretch")
        metric = st.selectbox("Bar-chart metric", list(table.columns),
                              index=list(table.columns).index("rms"))
        st.plotly_chart(plotly_bars(table, metric, conditions), width="stretch")
        st.download_button(
            "Download comparison table (CSV)",
            data=table_view.to_csv(index=True).encode("utf-8"),
            file_name="comparison_table.csv", mime="text/csv",
        )

    with tab_stats:
        st.subheader("Statistical significance across conditions")
        if not stats or stats.get("omnibus") is None or stats["omnibus"].empty:
            st.info("Significance tests need at least two conditions with enough "
                    "windows of data. Import a dataset with several conditions.")
        else:
            st.caption(f"{stats['n_windows']} analysis windows across "
                       f"{len(stats['conditions'])} conditions: {', '.join(stats['conditions'])}.")
            figure = plotly_pvalues(stats["omnibus"])
            if figure is not None:
                st.plotly_chart(figure, width="stretch")
            st.markdown("**Omnibus tests** (one-way ANOVA and Kruskal-Wallis):")
            st.dataframe(reporting.format_pvalues(stats["omnibus"]), width="stretch")
            st.markdown("**Pairwise Welch t-tests against the healthy baseline** "
                        "(Cohen's d = effect size):")
            pairwise = stats.get("pairwise")
            if pairwise is not None and not pairwise.empty:
                st.dataframe(reporting.format_pvalues(pairwise), width="stretch")
                notable = pairwise[pairwise["significant"] &
                                   pairwise["effect"].isin(["large", "medium"])]
                sig_features = stats.get("significant", [])
                if sig_features:
                    st.success("Features separating the conditions: " +
                               ", ".join(sig_features) + ".")
                if not notable.empty:
                    st.caption(f"{len(notable)} medium/large pairwise differences")
            else:
                st.info("No pairwise comparisons available.")

    with tab_sev:
        st.subheader("Fault severity (damage progression)")
        if severity_frame.empty:
            st.info("No standard CWRU damage-size codes (007/014/021) were found in the "
                    "recording names, so the severity trend is not available for this data.")
        else:
            st.caption("Damage size recovered from the CWRU fault codes "
                       "(0.007 / 0.014 / 0.021 inch).")
            metric = st.selectbox("Indicator", ["rms", "kurtosis", "crest_factor", "peak"],
                                  index=0, key="sev_metric")
            st.plotly_chart(plotly_severity(severity_frame, metric, conditions), width="stretch")
            st.markdown("**Recordings with a known fault size**")
            st.dataframe(severity_frame.round(4), width="stretch")
            st.markdown("**Trend vs damage size** (positive slope = grows with severity):")
            trends = severity.severity_trends(severity_frame,
                                              ["rms", "kurtosis", "crest_factor"])
            st.dataframe(trends.round(4), width="stretch")
            st.download_button(
                "Download severity table (CSV)",
                data=severity_frame.to_csv(index=False).encode("utf-8"),
                file_name="severity_table.csv", mime="text/csv",
            )

    with tab_report:
        st.subheader("Interpretation for machine-condition monitoring")
        for line in reporting.interpret(condition_table, stats=stats,
                                        severity_frame=severity_frame):
            st.markdown(f"- {line}")

        st.subheader("Generate results package")
        st.write("Creates the report and all figures (waveforms, spectra, RMS, "
                 "kurtosis, crest factor) and lets you download them.")
        if st.button("Generate report & figures", type="primary"):
            st.session_state["_results"] = build_results(processed, table, fs, signature, conditions)

        results = st.session_state.get("_results")
        if results and results.get("sig") == signature:
            d1, d2, d3 = st.columns(3)
            d1.download_button("⬇ Report (HTML)", results["html"], "report.html", "text/html")
            d2.download_button("⬇ Report (Markdown)", results["md"], "report.md", "text/markdown")
            d3.download_button("⬇ All results (ZIP)", results["zip"], "bearing_analysis_results.zip",
                               "application/zip")
            st.markdown("**Figures**")
            fig_cols = st.columns(3)
            for i, (name, blob) in enumerate(results["figures"].items()):
                fig_cols[i % 3].download_button(name, blob, name, "image/png", key=f"fig_{name}")
            with st.expander("Preview report (Markdown)"):
                st.code(results["md"].decode("utf-8"), language="markdown")
        elif results:
            st.info("Data changed — press the button to regenerate the results.")

    st.success("Analysis complete — explore the tabs above.")


if __name__ == "__main__":
    main()
