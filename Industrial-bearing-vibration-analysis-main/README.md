# Industrial Bearing Vibration Analysis

A modular Python toolkit for loading, preprocessing, and analysing vibration
signals from industrial rolling-element bearings. It extracts time-domain and
frequency-domain features, compares bearing health conditions statistically,
tracks fault severity, and visualises everything through an interactive
dashboard and a self-contained report.

The reference dataset is the **CWRU (Case Western Reserve University) Bearing
Data Center** dataset, but any vibration signal stored as CSV / MATLAB `.mat` /
NumPy `.npz` can be used.

---

## Table of contents

1. [What it does](#what-it-does)
2. [Requirements](#requirements)
3. [Quick start](#quick-start)
4. [Running the analysis (`main.py`)](#running-the-analysis-mainpy)
5. [Running the dashboard](#running-the-dashboard)
6. [Using your own data](#using-your-own-data)
7. [Outputs](#outputs)
8. [Project structure](#project-structure)
9. [Troubleshooting](#troubleshooting)
10. [Documentation](#documentation)

---

## What it does

- **Data loading** — read `.mat` / `.csv` / `.npz` recordings and attach
  condition labels (healthy, inner race, outer race, ball).
- **Preprocessing** — detrending, band-pass filtering, optional normalisation,
  and signal segmentation.
- **Time-domain features** — RMS, kurtosis, skewness, crest factor,
  peak-to-peak, shape and impulse factors, and more.
- **Frequency-domain features** — FFT spectrum, dominant frequencies, spectral
  centroid/bandwidth, and band-energy ratios.
- **Comparison** — per-recording feature table plus a condition-averaged
  summary.
- **Significance testing** — one-way ANOVA and Kruskal-Wallis across conditions,
  plus pairwise Welch t-tests against the healthy baseline with Cohen's d effect
  sizes.
- **Severity / damage progression** — tracks how indicators (RMS, kurtosis,
  crest factor) grow with the CWRU fault diameters (0.007 / 0.014 / 0.021 inch).
- **Reports** — a self-contained `report.html` and a `report.md` with figures,
  tables, statistics and an automatic interpretation.
- **Dashboard** — an interactive Streamlit app with six tabs.

---

## Requirements

- **Python** 3.9 or newer (3.10+ recommended).
- **pip** (bundled with Python).
- The libraries listed in [`requirements.txt`](requirements.txt):

  | Library      | Used for                                   |
  | ------------ | ------------------------------------------ |
  | numpy        | array math and FFT                         |
  | pandas       | feature tables                             |
  | scipy        | filtering, `.mat` loading, statistical tests |
  | matplotlib   | report figures                             |
  | streamlit    | interactive dashboard                      |
  | plotly       | dashboard charts                           |

---

## Quick start

### 1. Get the code

```bash
git clone https://github.com/deepika2517/Industrial-bearing-vibration-analysis.git
cd Industrial-bearing-vibration-analysis/Industrial-bearing-vibration-analysis-main
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv .venv

# Windows (PowerShell)
.venv\Scripts\Activate.ps1
# Windows (cmd)
.venv\Scripts\activate.bat
# macOS / Linux
source .venv/bin/activate
```

### 3. Install the dependencies

```bash
pip install -r requirements.txt
```

### 4. Run it

Two ways — pick either (or both):

```bash
# A) Full analysis: figures, tables and the HTML/Markdown report
python main.py --data data/raw --fs 48000

# B) Interactive dashboard (or just double-click run_dashboard.bat on Windows)
python -m streamlit run dashboard/app.py
```

> **No dataset yet?** Just run `python main.py` with no `--data` argument — it
> generates synthetic demo signals so you can see the full output immediately.

---

## Running the analysis (`main.py`)

```bash
python main.py [--data DIR] [--fs HZ] [--duration SEC] [--output DIR]
```

| Argument     | Default      | Meaning                                                        |
| ------------ | ------------ | ------------------------------------------------------------- |
| `--data`     | *(none)*     | Folder with recordings; omit for synthetic demo signals.      |
| `--fs`       | `12000`      | Sampling frequency in Hz. **Match your dataset.**             |
| `--duration` | `1.0`        | Synthetic demo signal length (seconds).                       |
| `--output`   | `results`    | Folder for figures, tables and the report.                    |

Examples:

```bash
python main.py                                   # synthetic demo -> results/
python main.py --data data/raw --fs 48000        # CWRU 48 kHz drive-end data
python main.py --data my_data --fs 12000 --output out
```

Every run writes the figures, `comparison_table.csv`, `condition_summary.csv`
and the `report.html` / `report.md` into the output folder.

---

## Running the dashboard

On **Windows you can double-click `run_dashboard.bat`** — no terminal needed.
Otherwise run:

```bash
python -m streamlit run dashboard/app.py
```

A browser tab opens at `http://localhost:8501`. In the **sidebar**, choose where
the data comes from:

- **Upload files** — drag in `.mat` / `.csv` / `.npz` recordings. The condition
  is inferred from the filename (`IR`→inner race, `OR`→outer race, `B`→ball,
  `Normal`→healthy) and can be corrected afterwards.
- **Folder path(s)** — list one or more folder paths (one per line), or use the
  **Browse** button to pick folders with a native dialog. All folders are merged
  into one dataset.
- **Built-in demo** — synthetic signals, works with no data.

Then explore the six tabs:

| Tab                    | What it shows                                                       |
| ---------------------- | ------------------------------------------------------------------- |
| **Raw Waveform**       | The selected recording and its time-domain features.                |
| **FFT Spectrum**       | Overlaid spectra and the dominant frequencies of selected records.  |
| **Feature Comparison** | Condition summary + per-recording feature tables and a bar chart.   |
| **Statistics**         | ANOVA / Kruskal-Wallis tests and pairwise t-tests vs. healthy.      |
| **Severity**           | Indicator growth against damage size, with a trend table.           |
| **Results & Report**   | Interpretation + one-click download of the report, figures and ZIP. |

---

## Using your own data

- **Supported formats:** MATLAB `.mat`, comma-separated `.csv`, NumPy `.npz`.
- **Condition labels:** put recordings in folders named after the condition
  (`healthy`, `inner_race`, `outer_race`, `ball`) **or** use CWRU-style
  filenames (`Normal`, `IR007`, `OR014`, `B021`, …). The label is inferred
  automatically and can be corrected in the dashboard.
- **Sampling frequency:** set `--fs` (or the sidebar field) to match your data.
  The 48 kHz CWRU Drive-End files must use `--fs 48000`; the 12 kHz files use
  `--fs 12000`.
- **Fault size:** recordings whose name contains a CWRU code (`007`, `014`,
  `021`) automatically feed the severity analysis.

Expected layout for a multi-condition run:

```
data/raw/
├── healthy/        Normal_1_098.mat
├── inner_race/     IR007_1_110.mat  IR014_1_175.mat  IR021_1_214.mat
├── outer_race/     OR007_6_136.mat  OR014_6_202.mat  OR021_6_239.mat
└── ball/           B007_1_123.mat   B014_1_190.mat   B021_1_227.mat
```

See [`data/README.md`](data/README.md) for where to download the CWRU data.

---

## Outputs

After a run of `main.py` (or "Generate report & figures" in the dashboard):

| File                     | Contents                                              |
| ------------------------ | ----------------------------------------------------- |
| `report.html`            | Self-contained report (figures embedded).             |
| `report.md`              | Markdown version of the report.                       |
| `comparison_table.csv`   | Per-recording feature table.                          |
| `condition_summary.csv`  | Feature table averaged per condition.                 |
| `waveforms.png`          | Raw waveforms by condition.                           |
| `spectra_comparison.png` | Overlaid FFT amplitude spectra.                       |
| `rms_comparison.png`     | RMS by condition.                                     |
| `kurtosis_comparison.png`| Kurtosis by condition.                                |
| `crest_factor_comparison.png` | Crest factor by condition.                      |

---

## Project structure

```
industrial-bearing-vibration-analysis/
├── data/                     # Dataset instructions + small sample files
│   ├── samples/
│   └── README.md
├── src/                      # Analysis modules
│   ├── data_loading.py       # read .mat / .csv / .npz, infer conditions
│   ├── preprocessing.py      # detrend, band-pass filter, normalise
│   ├── statistical_analysis.py  # time-domain indicators
│   ├── fft_analysis.py       # frequency-domain features
│   ├── comparison.py         # comparison tables + figures
│   ├── statistical_tests.py  # ANOVA / Kruskal-Wallis / pairwise t-tests
│   ├── severity.py           # damage-size (severity) trends
│   └── reporting.py          # HTML / Markdown report generation
├── dashboard/
│   └── app.py                # Streamlit dashboard
├── results/                  # Generated figures, tables and reports
├── docs/                     # Methodology and project report
│   ├── methodology.md
│   └── report.md
├── main.py                   # Runs the analysis pipeline
├── run_dashboard.bat         # Double-click launcher (Windows)
├── requirements.txt          # Python dependencies
├── README.md                 # This file
└── .gitignore
```

---

## Troubleshooting

| Problem                                        | Fix                                                                 |
| ---------------------------------------------- | ------------------------------------------------------------------- |
| `streamlit: command not found`                 | Use `python -m streamlit run dashboard/app.py` instead.             |
| Dominant frequencies look wrong / too high-low | Set `--fs` (or the sidebar value) to the dataset's real sample rate.|
| `No recordings found in those folders`         | Check the path; only `.mat`, `.csv`, `.npz` files are read.         |
| A `.mat` file loads the wrong signal           | Files with stray variables are handled by preferring `*_DE_time`; rename to include the CWRU number if needed. |
| Dashboard says "Add at least one folder path"  | Enter a valid folder, or switch to **Built-in demo**.               |

---

## Documentation

- [`docs/methodology.md`](docs/methodology.md) — pipeline, feature definitions,
  statistical methods, severity analysis, and limitations.
- [`docs/report.md`](docs/report.md) — full project report with results and
  discussion.
- [`data/README.md`](data/README.md) — dataset download and layout.

---

## License

Provided for educational and research purposes.
