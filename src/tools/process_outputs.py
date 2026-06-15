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


def writeTargetGeometry(cfg_path, out_path):
    """Per-board geometry REFERENCE table (one row per cell): position in the grid and on
    the metric board plane (mm), and the reach distance from the participant entry side
    (bottom-centre). It is a property of the BOARD, identical for every participant, so it
    is published ONCE here instead of being repeated on every trial row. Join it to the
    trials by trial_name (= color_shape of the target piece) or by target_row/target_col."""
    import csv as _csv
    import math as _math
    import yaml as _yaml
    cfg = _yaml.safe_load(open(cfg_path))
    bs, bs_mm, board = cfg['board_size'], cfg['board_size_mm'], cfg['board_config']
    rows = [['trial_name', 'color', 'shape', 'is_piece', 'target_row', 'target_col',
             'target_x_mm', 'target_y_mm', 'reach_distance_mm']]
    for key, (color, shape, is_piece) in board.items():
        col, row = (int(x) for x in key.split(','))    # cfg key is 'col,row'
        tx = (col + 0.5) / bs[0] * bs_mm[0]
        ty = (row + 0.5) / bs[1] * bs_mm[1]
        dist = round(_math.hypot(tx - bs_mm[0] / 2.0, ty - bs_mm[1]), 1)
        rows.append([f'{color}_{shape}', color, shape, is_piece, row, col,
                     round(tx, 1), round(ty, 1), dist])
    with open(out_path, 'w', newline='') as f:
        _csv.writer(f).writerows(rows)
    return len(rows) - 1


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
        n3 = combineCsvs(topic_dir, 'trials_data_{}_transitions.csv', os.path.join(out_dir, f'combined_transitions_{args.topic}.csv'))
        print(f"Combined trials CSV: {n1} participants")
        print(f"Combined sequence CSV: {n2} participants")
        print(f"Combined transitions CSV: {n3} participants")

    # 2b. Per-board geometry REFERENCE table (one row per cell; same for every
    # participant). Published once so the per-trial CSV need not repeat reach distance.
    cfg_path = os.path.join(REPO_ROOT, 'cfg', 'game_config.yaml')
    if os.path.isfile(cfg_path):
        n_geo = writeTargetGeometry(cfg_path, os.path.join(out_dir, 'target_geometry.csv'))
        print(f"Target geometry CSV: {n_geo} cells")

    # 3-4. HTML report + frequencies CSV (generate_report writes both)
    report_argv = ['--roots', *roots, '--data_root', args.data_root, '-t', args.topic,
                   '--out', os.path.join(out_dir, 'informe_comparativa.html')]
    sys.argv = ['generate_report'] + report_argv
    report.main()

    print(f"\nDone. All artefacts in {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
