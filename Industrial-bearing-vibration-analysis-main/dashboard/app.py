
"""Streamlit dashboard for industrial bearing vibration analysis."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import data_loading, preprocessing, statistical_analysis, fft_analysis


@st.cache_data(show_spinner=False)
def load_real_data(data_path, fs):
    signals = data_loading.load_directory(data_path, fs=fs)
    return {
        name: {
            "signal": item.signal,
            "condition": item.condition,
            "fs": item.fs,
        }
        for name, item in signals.items()
    }


def main():
    st.set_page_config(
        page_title="Bearing Vibration Analysis",
        page_icon="📊",
        layout="wide",
    )

    st.title("Industrial Bearing Vibration Analysis")
    st.caption("CWRU vibration data — time-domain and frequency-domain analysis")

    with st.sidebar:
        st.header("Settings")
        data_path = st.text_input(
            "Dataset folder",
            value=str(ROOT / "data" / "raw"),
        )
        fs = st.number_input(
            "Sampling frequency (Hz)",
            min_value=100.0,
            value=12000.0,
            step=1000.0,
        )
        max_samples = st.number_input(
            "Waveform samples to display",
            min_value=500,
            max_value=10000,
            value=2000,
            step=500,
        )

    if not Path(data_path).exists():
        st.error(f"Dataset folder not found: {data_path}")
        st.stop()

    with st.spinner("Loading vibration recordings..."):
        data = load_real_data(data_path, fs)

    if not data:
        st.error("No CSV or MAT recordings could be loaded from this folder.")
        st.info("Check that your recordings are inside data/raw and use .mat or .csv extensions.")
        st.stop()

    names = list(data.keys())
    selected_name = st.selectbox("Select a recording", names)
    selected = data[selected_name]
    raw_signal = np.asarray(selected["signal"], dtype=float)

    if raw_signal.size == 0:
        st.error("The selected recording contains no samples.")
        st.stop()

    # Keep the original signal for raw measurements.
    processed = preprocessing.preprocess_pipeline(
        raw_signal, fs, normalize_method=None
    )

    features = statistical_analysis.time_domain_features(processed)

    st.subheader("Selected recording")
    m1, m2, m3 = st.columns(3)
    m1.metric("Condition", selected["condition"])
    m2.metric("Samples", f"{len(raw_signal):,}")
    m3.metric("Duration", f"{len(raw_signal) / fs:.2f} s")

    left, right = st.columns(2)

    with left:
        st.subheader("Vibration waveform")
        count = min(int(max_samples), len(raw_signal))
        time = np.arange(count) / fs
        st.line_chart(
            pd.DataFrame(
                {"Amplitude": raw_signal[:count]},
                index=pd.Index(time, name="Time (s)"),
            )
        )

    with right:
        st.subheader("Frequency spectrum")
        freqs, magnitude = fft_analysis.compute_fft(processed, fs)
        limit = min(len(freqs), 5000)
        spectrum = pd.DataFrame(
            {"Magnitude": magnitude[:limit]},
            index=pd.Index(freqs[:limit], name="Frequency (Hz)"),
        )
        st.line_chart(spectrum)

    st.subheader("Time-domain features")
    st.dataframe(
        pd.DataFrame([features], index=[selected_name]).round(5),
        use_container_width=True,
    )

    st.subheader("Comparison across recordings")
    rows = {}
    for name, item in data.items():
        signal = preprocessing.preprocess_pipeline(
            item["signal"], fs, normalize_method=None
        )
        rows[name] = statistical_analysis.time_domain_features(signal)

    table = pd.DataFrame.from_dict(rows, orient="index")
    st.dataframe(table.round(5), use_container_width=True)

    st.subheader("RMS Comparison")
    st.bar_chart(table[["rms"]])

    st.subheader("Kurtosis Comparison")
    st.bar_chart(table[["kurtosis"]])

    st.subheader("Crest Factor Comparison")
    st.bar_chart(table[["crest_factor"]])

    st.download_button(
        "Download feature comparison (CSV)",
        data=table.to_csv(index=True).encode("utf-8"),
        file_name="bearing_feature_comparison.csv",
        mime="text/csv",
    )

    st.success("Analysis complete. Results are based on the loaded recordings.")


if __name__ == "__main__":
    main()
