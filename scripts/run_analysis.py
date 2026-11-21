#!/usr/bin/env python3
"""Prepare the source data and run a bounded, reproducible benchmark."""
from pathlib import Path
import argparse
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.traffic_analysis import ROOT, PREPARED, prepare, load_prepared, plot_eda, benchmark, plot_evaluation

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw', type=Path, help='Path to the downloaded Traffic_Violations.csv')
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--skip-prepare', action='store_true')
    parser.add_argument('--max-stops', type=int, default=200_000)
    args = parser.parse_args()
    if args.max_stops < 1000:
        parser.error('--max-stops must be at least 1000')
    if not args.skip_prepare:
        print(prepare(args.raw), flush=True)
    if args.prepare_only:
        return
    import matplotlib
    matplotlib.use('Agg')
    data = load_prepared()
    import matplotlib.pyplot as plt
    plt.close(plot_eda(data))
    results, validation, metadata, y, scores = benchmark(data, args.max_stops)
    plt.close(plot_evaluation(results, metadata, y, scores))
    print(results.to_string(index=False))

if __name__ == '__main__':
    main()
