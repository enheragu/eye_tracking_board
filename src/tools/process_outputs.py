#!/usr/bin/env python3
# encoding: utf-8

"""
    One entry point that produces every cross-participant / cross-version artefact for
    the research team, so there is a single command to run after processing:
      - combined per-trial CSV (all participants stacked, participant as first column)
      - combined gaze-sequence CSV (all participants stacked)
      - per-participant frequencies CSV (world fps, gaze rate, continuity)
      - HTML comparison report (tabs + version checkboxes)

    Examples:
        # Default: latest output root vs the previous versioned roots found next to it
        python3 src/tools/process_outputs.py

        # Explicit roots and output folder
        python3 src/tools/process_outputs.py \
            --roots /path/OutpuData_v0.8.0 /path/OutputData_v1.0.0 /path/OutputData_v1.1.0 \
            --data_root /path/InputData --out_dir /path/reports
"""

import os
import sys
import glob
import argparse

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)

from src.core.version import __version__
from src.tools.combine_outputs import combineCsvs
from src.tools import generate_report as report

DEFAULT_DATA_ROOT = os.environ.get('EEHA_DATA_ROOT', '/media/quique/EXTERNAL_USB1/BusquedaVisualAnalysis/InputData')
DEFAULT_BASE = os.path.dirname(DEFAULT_DATA_ROOT)


def discoverVersionRoots(base):
    """All OutputData_v* roots under base, oldest first (also the historical typo)."""
    roots = glob.glob(os.path.join(base, 'Out*Data_v*'))
    return sorted(r for r in roots if os.path.isdir(r))


def main():
    parser = argparse.ArgumentParser(description='Produce all team artefacts from the processing outputs')
    parser.add_argument('--roots', nargs='+', default=None,
                        help='Output roots to compare (oldest first). Default: all OutputData_v* next to the data root.')
    parser.add_argument('--data_root', default=DEFAULT_DATA_ROOT, help='Input data root (for frequencies).')
    parser.add_argument('-t', dest='topic', default='gaze')
    parser.add_argument('--out_dir', default=None, help='Where to write the artefacts. Default: the latest root.')
    args = parser.parse_args()

    roots = args.roots or discoverVersionRoots(DEFAULT_BASE)
    if not roots:
        print(f"No output roots found. Provide --roots.")
        return 1
    latest = roots[-1]
    out_dir = args.out_dir or latest
    os.makedirs(out_dir, exist_ok=True)

    print(f"Versions: {[report.versionLabel(r) for r in roots]}")
    print(f"Artefacts -> {out_dir}\n")

    # 1-2. Combined CSVs (stacked participants) of the LATEST version
    topic_dir = os.path.join(latest, args.topic)
    if os.path.isdir(topic_dir):
        n1 = combineCsvs(topic_dir, 'trials_data_{}.csv', os.path.join(out_dir, f'combined_trials_{args.topic}.csv'))
        n2 = combineCsvs(topic_dir, 'trials_data_{}_sequence.csv', os.path.join(out_dir, f'combined_sequence_{args.topic}.csv'))
        print(f"Combined trials CSV: {n1} participants")
        print(f"Combined sequence CSV: {n2} participants")

    # 3-4. HTML report + frequencies CSV (generate_report writes both)
    report_argv = ['--roots', *roots, '--data_root', args.data_root, '-t', args.topic,
                   '--out', os.path.join(out_dir, 'informe_comparativa.html')]
    sys.argv = ['generate_report'] + report_argv
    report.main()

    print(f"\nDone. All artefacts in {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
