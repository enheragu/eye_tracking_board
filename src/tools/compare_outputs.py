#!/usr/bin/env python3
# encoding: utf-8

"""
    Compares two output roots (e.g. results produced by SW v0.x against v1.x) so the
    impact of processing changes can be reviewed at a glance: how many valid trials
    each version segmented, which errored trials were rescued, and how trial
    durations shifted.

    Examples:
        # Summary table over all participants found in both roots
        python3 src/tools/compare_outputs.py --old /path/old_output --new /path/new_output

        # Per-trial detail for one participant
        python3 src/tools/compare_outputs.py --old ... --new ... -p 002 --detail
"""

import os
import csv
import argparse
import statistics

from tabulate import tabulate

DEFAULT_NEW_ROOT = os.environ.get('EEHA_OUTPUT_ROOT', '/media/quique/EXTERNAL_USB1/BusquedaVisualAnalysis/OutputData')

ERROR_PREFIXES = ('missing_trial_error', 'transition_error', 'end_of_video_error')


def loadTrials(csv_path):
    trials = {}
    if not os.path.isfile(csv_path):
        return None
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            key = (int(row['block_index']), int(row['trial_index']))
            trial = trials.setdefault(key, {'name': row['trial_name'],
                                            'duration': float(row['trial_duration_s']),
                                            'status': row['Finish Status'],
                                            'piece': 0, 'slot': 0})
            trial['piece'] += int(row['Piece Fixations'])
            trial['slot'] += int(row['Slot only Fixations'])
    return trials


def isValid(trial):
    return trial is not None and not trial['name'].startswith(ERROR_PREFIXES)


def compareParticipant(participant, old_trials, new_trials):
    keys = set(old_trials) | set(new_trials)
    valid_old = sum(1 for k in keys if isValid(old_trials.get(k)))
    valid_new = sum(1 for k in keys if isValid(new_trials.get(k)))
    rescued = sum(1 for k in keys if not isValid(old_trials.get(k)) and isValid(new_trials.get(k)))
    lost = sum(1 for k in keys if isValid(old_trials.get(k)) and not isValid(new_trials.get(k)))
    closed_by_panel = sum(1 for k in keys if isValid(new_trials.get(k))
                          and new_trials[k]['status'] == 'test_finish_by_next_panel')

    duration_diffs = [new_trials[k]['duration'] - old_trials[k]['duration']
                      for k in keys if isValid(old_trials.get(k)) and isValid(new_trials.get(k))]
    median_diff = statistics.median(duration_diffs) if duration_diffs else float('nan')
    big_changes = sum(1 for d in duration_diffs if abs(d) > 1.0)

    return [participant, valid_old, valid_new, rescued, lost, closed_by_panel,
            f"{median_diff:+.3f}", big_changes]


def detailParticipant(participant, old_trials, new_trials):
    keys = sorted(set(old_trials) | set(new_trials))
    table = []
    for key in keys:
        old, new = old_trials.get(key), new_trials.get(key)
        diff = f"{new['duration']-old['duration']:+.2f}" if old and new else '-'
        table.append([f"{key[0]},{key[1]}",
                      old['name'] if old else '--missing--',
                      new['name'] if new else '--missing--',
                      f"{old['duration']:.2f}" if old else '-',
                      f"{new['duration']:.2f}" if new else '-',
                      diff,
                      (old['piece'] + old['slot']) if old else '-',
                      (new['piece'] + new['slot']) if new else '-',
                      new['status'] if new else '-'])
    headers = ['blk,tr', 'old name', 'new name', 'old s', 'new s', 'diff s',
               'old fix', 'new fix', 'new status']
    print(f"\n=== Participant {participant} ===")
    print(tabulate(table, headers=headers, tablefmt='pretty'))


def main():
    parser = argparse.ArgumentParser(description='Compare two processing output roots')
    parser.add_argument('--old', required=True, help='Output root of the previous version.')
    parser.add_argument('--new', default=DEFAULT_NEW_ROOT, help='Output root of the new version.')
    parser.add_argument('-t', dest='topic', type=str, default='gaze', help='Eye data topic subfolder.')
    parser.add_argument('-p', '--participants', nargs='*', default=None,
                        help='Participants to compare (default: all present in both roots).')
    parser.add_argument('--detail', action='store_true', help='Print the per-trial table of each participant.')
    args = parser.parse_args()

    old_root = os.path.join(args.old, args.topic)
    new_root = os.path.join(args.new, args.topic)

    if args.participants:
        participants = args.participants
    else:
        old_ids = set(os.listdir(old_root)) if os.path.isdir(old_root) else set()
        new_ids = set(os.listdir(new_root)) if os.path.isdir(new_root) else set()
        participants = sorted(old_ids & new_ids)

    if not participants:
        print(f"No common participants found between {old_root} and {new_root}")
        return 1

    summary = []
    for participant in participants:
        old_trials = loadTrials(os.path.join(old_root, participant, f'trials_data_{participant}.csv'))
        new_trials = loadTrials(os.path.join(new_root, participant, f'trials_data_{participant}.csv'))
        if old_trials is None or new_trials is None:
            print(f"[WARN] Missing trials_data CSV for participant {participant}, skipped.")
            continue
        summary.append(compareParticipant(participant, old_trials, new_trials))
        if args.detail:
            detailParticipant(participant, old_trials, new_trials)

    headers = ['Participant', 'Valid old', 'Valid new', 'Rescued', 'Lost',
               'Closed by panel', 'Median dur diff (s)', '|diff|>1s']
    print("\n=== Output comparison summary ===")
    print(tabulate(summary, headers=headers, tablefmt='pretty'))
    print("\n· 'Rescued': trials errored in the old output that are valid in the new one.")
    print("· 'Closed by panel': valid trials with status test_finish_by_next_panel.")
    print("· Median duration diff ~ -0.17s is expected (end backdating removes the")
    print("  occlusion-confirmation overhead). Trials with |diff|>1s deserve a look")
    print("  with the debug video (-v).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
