# Industrial Bearing Vibration Analysis

A modular Python toolkit for loading, preprocessing, and analysing vibration
signals from industrial rolling-element bearings. It extracts time-domain and
frequency-domain features, compares bearing health conditions, and visualises
the results through an interactive dashboard.

The reference dataset is the **CWRU (Case Western Reserve University) Bearing
Data Center** dataset, but any vibration signal stored as CSV or MATLAB `.mat`
files can be used.

## Features

- **Data loading** — read CSV / `.mat` recordings and attach condition labels.
- **Preprocessing** — detrending, normalisation, band-pass filtering, and
  signal segmentation.
- **Statistical analysis** — RMS, kurtosis, skewness, crest factor, peak-to-peak,
  and other time-domain indicators.
- **FFT analysis** — frequency spectrum, dominant frequencies, spectral
  centroid, and band energy ratios.
- **Comparison** — side-by-side comparison of healthy vs. faulty bearings.
- **Dashboard** — an interactive Streamlit app to explore signals and features.

## Project structure

```
industrial-bearing-vibration-analysis/
├── data/               # Dataset instructions and small sample files
│   ├── samples/
│   └── README.md
├── src/                # Main Python analysis modules
│   ├── data_loading.py
│   ├── preprocessing.py
│   ├── statistical_analysis.py
│   ├── fft_analysis.py
│   └── comparison.py
├── dashboard/          # Interactive dashboard
│   └── app.py
├── results/            # Generated graphs, comparison tables and results
├── docs/               # Project report, methodology and documentation
├── main.py             # Runs the analysis
├── requirements.txt    # Required Python libraries
├── README.md           # Project overview and setup instructions
└── .gitignore          # Files Git should not upload
```

## Setup

```bash
# Clone the repository
git clone https://github.com/deepika2517/Industrial-bearing-vibration-analysis.git
cd Industrial-bearing-vibration-analysis

# Create and activate a virtual environment
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

Run the full analysis pipeline (generates a demo signal if no dataset is
present, then writes figures and tables to `results/`):

```bash
python main.py
```

Run with your own dataset:

```bash
python main.py --data data/raw --fs 12000
```

Launch the interactive dashboard:

```bash
streamlit run dashboard/app.py
```

## Data

See [`data/README.md`](data/README.md) for download instructions and the
expected file layout. Small synthetic sample recordings live in `data/samples/`.

## Documentation

Methodology, feature definitions, and the project report live in [`docs/`](docs/).

## License

Provided for educational and research purposes.
