#!/usr/bin/env python3
# encoding: utf-8

"""
    Runs process_video.py for several participants in parallel (one subprocess per
    participant). Jobs and OpenCV threads per job are bounded so the machine is not
    oversubscribed: with the defaults, jobs * cv_threads ~= number of cores.

    Examples:
        python3 run_all.py                      # all participants found in data root
        python3 run_all.py -p 002 024 -j 2      # only some participants, 2 at a time
"""

import os
import sys
import time
import argparse
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed

CURRENT_FILE_PATH = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA_ROOT = os.environ.get('EEHA_DATA_ROOT', '/media/quique/EXTERNAL_USB1/BusquedaVisualAnalysis/InputData')
DEFAULT_OUTPUT_ROOT = os.environ.get('EEHA_OUTPUT_ROOT', '/media/quique/EXTERNAL_USB1/BusquedaVisualAnalysis/OutputData')


def discoverParticipants(data_root):
    if not os.path.isdir(data_root):
        raise FileNotFoundError(f"Data root not found: {data_root}. Is the drive mounted?")
    participants = []
    for name in sorted(os.listdir(data_root)):
        if os.path.isfile(os.path.join(data_root, name, 'world.mp4')):
            participants.append(name)
    return participants


def runParticipant(participant, args, env):
    output_dir = os.path.join(args.output_root, args.topic, participant)
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, 'process_log.txt')

    cmd = [sys.executable, os.path.join(CURRENT_FILE_PATH, 'process_video.py'),
           '-p', participant, '-t', args.topic,
           '--data_root', args.data_root, '--output_root', args.output_root]
    if not args.fast_analysis:
        cmd.append('--slow_analysis')

    start = time.time()
    with open(log_path, 'w') as log_file:
        result = subprocess.run(cmd, stdout=log_file, stderr=subprocess.STDOUT, env=env)
    return participant, result.returncode, time.time() - start, log_path


def main():
    parser = argparse.ArgumentParser(description='Process all participants in parallel')
    parser.add_argument('-p', '--participants', nargs='*', default=None,
                        help='Participant ids to process. By default all folders with a world.mp4 in the data root.')
    parser.add_argument('-t', dest='topic', type=str, default='gaze', help='Eye data topic to process (gaze/fixations).')
    parser.add_argument('-j', '--jobs', type=int, default=max(1, (os.cpu_count() or 4) // 2),
                        help='Simultaneous participants to process.')
    parser.add_argument('--cv_threads', type=int, default=2, help='OpenCV threads for each job.')
    parser.add_argument('--fast_analysis', action='store_true', default=False,
                        help='Disable --slow_analysis (faster, less precise transition detection).')
    parser.add_argument('--data_root', type=str, default=DEFAULT_DATA_ROOT)
    parser.add_argument('--output_root', type=str, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args()

    participants = args.participants if args.participants else discoverParticipants(args.data_root)
    if not participants:
        print(f"No participants found in {args.data_root}")
        return 1

    env = os.environ.copy()
    env['EEHA_CV_THREADS'] = str(args.cv_threads)

    print(f"Processing {len(participants)} participants with {args.jobs} jobs "
          f"({args.cv_threads} OpenCV threads each): {participants}")

    results = []
    try:
        with ThreadPoolExecutor(max_workers=args.jobs) as executor:
            futures = {executor.submit(runParticipant, participant, args, env): participant
                       for participant in participants}
            for future in as_completed(futures):
                participant, returncode, elapsed, log_path = future.result()
                status = 'OK' if returncode == 0 else f'ERROR ({returncode})'
                print(f"[{time.strftime('%H:%M:%S')}] [{status}] {participant} "
                      f"in {elapsed/60:.1f} min. Log: {log_path}")
                results.append((participant, returncode))
    except KeyboardInterrupt:
        print("\nInterrupted. Already-finished participants are kept; running ones may be incomplete.")
        return 130

    failed = [p for p, code in results if code != 0]
    print(f"\nFinished: {len(results) - len(failed)} OK, {len(failed)} with errors.")
    if failed:
        print(f"Failed participants: {failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
