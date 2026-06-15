#!/usr/bin/env python3
# encoding: utf-8

"""
    Stacks the per-participant CSVs of an output root into a single combined CSV with
    the participant id as the first column, so all participants can be analysed from
    one file. Combines both the per-trial summary and the gaze sequence CSVs.

    Examples:
        python3 src/tools/combine_outputs.py                       # default output root
        python3 src/tools/combine_outputs.py --root /path/Output   # a specific root
"""

import os
import sys
import csv
import glob
import argparse

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)
from src.core.version import __version__

DEFAULT_OUTPUT_ROOT = os.environ.get('EEHA_OUTPUT_ROOT', f'/media/quique/EXTERNAL_USB1/BusquedaVisualAnalysis/OutputData_v{__version__}')


def combineCsvs(topic_dir, filename_tpl, out_path):
    """Stacks every participant CSV (filename_tpl with the id) under topic_dir,
    prepending a 'participant' column. Returns the number of participants combined."""
    participants = sorted(d for d in os.listdir(topic_dir) if os.path.isdir(os.path.join(topic_dir, d)))
    header_written = False
    n = 0
    with open(out_path, 'w', newline='') as out_file:
        writer = csv.writer(out_file)
        for participant in participants:
            csv_path = os.path.join(topic_dir, participant, filename_tpl.format(participant))
            if not os.path.isfile(csv_path):
                continue
            with open(csv_path) as in_file:
                reader = csv.reader(in_file)
                rows = list(reader)
            if not rows:
                continue
            if not header_written:
                writer.writerow(['participant'] + rows[0])
                header_written = True
            for row in rows[1:]:
                writer.writerow([participant] + row)
            n += 1
    return n


def main():
    parser = argparse.ArgumentParser(description='Combine per-participant CSVs into one file')
    parser.add_argument('--root', default=DEFAULT_OUTPUT_ROOT, help='Output root to combine.')
    parser.add_argument('-t', dest='topic', default='gaze', help='Eye data topic subfolder.')
    parser.add_argument('--out_dir', default=None, help='Where to write the combined CSVs (default: the root).')
    args = parser.parse_args()

    topic_dir = os.path.join(args.root, args.topic)
    if not os.path.isdir(topic_dir):
        print(f"Topic folder not found: {topic_dir}")
        return 1
    out_dir = args.out_dir or args.root
    os.makedirs(out_dir, exist_ok=True)

    trials_out = os.path.join(out_dir, f'combined_trials_{args.topic}.csv')
    seq_out = os.path.join(out_dir, f'combined_sequence_{args.topic}.csv')
    trans_out = os.path.join(out_dir, f'combined_transitions_{args.topic}.csv')
    n1 = combineCsvs(topic_dir, 'trials_data_{}.csv', trials_out)
    n2 = combineCsvs(topic_dir, 'trials_data_{}_sequence.csv', seq_out)
    n3 = combineCsvs(topic_dir, 'trials_data_{}_transitions.csv', trans_out)

    print(f"Combined per-trial summary of {n1} participants -> {trials_out}")
    print(f"Combined gaze sequence of {n2} participants -> {seq_out}")
    print(f"Combined state transitions of {n3} participants -> {trans_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
