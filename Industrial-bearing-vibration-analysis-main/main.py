"""Entry point: run the bearing vibration analysis pipeline.

Usage:
    python main.py
    python main.py --data data/raw --fs 12000
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src import comparison, data_loading, preprocessing, reporting
from src.data_loading import VibrationSignal

RESULTS_DIR = Path(__file__).resolve().parent / "results"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Industrial bearing vibration analysis")
    parser.add_argument("--data", type=str, default=None,
                        help="Directory containing CSV/.mat recordings. "
                             "If omitted, synthetic demo signals are used.")
    parser.add_argument("--fs", type=float, default=12000.0,
                        help="Sampling frequency in Hz (default: 12000).")
    parser.add_argument("--duration", type=float, default=1.0,
                        help="Demo signal duration in seconds (default: 1.0).")
    parser.add_argument("--output", type=str, default=str(RESULTS_DIR),
                        help="Output directory for figures and tables.")
    return parser.parse_args()


def load_signals(args: argparse.Namespace):
    """Load real data when a directory is provided, else build demo signals."""
    if args.data:
        print(f"[main] loading recordings from {args.data}")
        signals = data_loading.load_directory(args.data, fs=args.fs)
        if not signals:
            print("[main] no recordings found — falling back to synthetic demo data")
            signals = data_loading.make_demo_dataset(fs=args.fs, duration=args.duration)
        return signals
    print("[main] no --data given — using synthetic demo signals")
    return data_loading.make_demo_dataset(fs=args.fs, duration=args.duration)



def preprocess_all(signals, fs: float):
    """Preprocess signals without z-score normalisation."""
    preprocessed = {}
    for name, sig in signals.items():
        signal = sig.signal if isinstance(sig, VibrationSignal) else sig
        preprocessed[name] = preprocessing.preprocess_pipeline(
            signal, fs, normalize_method=None
        )
    return preprocessed



def main() -> None:
    args = parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    signals = load_signals(args)
    print(f"[main] loaded {len(signals)} signal(s): {', '.join(signals)}")

    processed = preprocess_all(signals, args.fs)

    table = comparison.build_comparison_table(processed, args.fs)
    table_path = comparison.save_comparison(table, output / "comparison_table.csv")
    print(f"[main] comparison table -> {table_path}")
    print(table.round(4).to_string())

    comparison.plot_waveforms(processed, args.fs, save_path=output / "waveforms.png")
    comparison.plot_comparison(processed, args.fs, metric="rms",
                               save_path=output / "rms_comparison.png")
    comparison.plot_comparison(processed, args.fs, metric="kurtosis",
                               save_path=output / "kurtosis_comparison.png")
    comparison.plot_spectra(processed, args.fs, save_path=output / "spectra_comparison.png")
    print(f"[main] figures written to {output}")

    report = reporting.generate_report(
        processed, args.fs, output,
        conditions=data_loading.condition_map(signals),
    )
    print(f"[main] report (HTML)     -> {report['html']}")
    print(f"[main] report (Markdown) -> {report['markdown']}")
    print("[main] done.")


if __name__ == "__main__":
    main()
