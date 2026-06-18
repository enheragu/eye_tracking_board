#!/usr/bin/env python3
# encoding: utf-8
"""Re-run ONLY the post-hoc landmark stage on ALREADY-PROCESSED participants, WITHOUT
re-decoding the video.

The expensive part of the pipeline is watching the video (ArUco detection + colour
correction, ~20-25 min for the 22). Everything the post-hoc landmark model needs is the
per-frame occlusion profile, which is already persisted in each data_<id>.pkl as the
`signal_trace` (target fT, whole-board board_occ, contour, homography, grid, per frame).
This tool reloads that, RE-APPLIES the exact StateMachine post-hoc stage (bump model:
target_touch = target peak, hand_exit = board valley adaptive, motor_onset contour
validated-by-occlusion hybrid, temporal congruence + reach_style) and regenerates the PKL
and CSVs. So a tweak to _posthocBump / its thresholds can be tested on all 22 in SECONDS.

Single source of truth: it imports and calls the real StateMachine._posthocBump and
store_results -- it does NOT re-implement any landmark logic.

Usage:
  reprocess_landmarks.py -p all  --input_root .../OutputData_v1.3.0 --output_root .../OutputData_v1.3.1
  reprocess_landmarks.py -p 049 --input_root ... --output_root ... --dry_run   # only report changes
"""
import os
import sys
import types
import pickle
import argparse

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)

from src.core.utils import log
from src.core.BoardHandler import BoardHandler
from src.core.PanelHandler import PanelHandler
from src.core.DistortionHandler import DistortionHandler
from src.core.StateMachineHandler import StateMachine

# Same colour configuration as process_video.py (needed only to construct the handlers).
colors_dict = {'red':   {'h': 350, 'eps': 29}, 'green':  {'h': 125, 'eps': 35},
               'blue':  {'h': 220, 'eps': 35}, 'yellow': {'h': 50,  'eps': 28}}
colors_list = {'red': (0, 0, 255), 'green': (0, 255, 0), 'blue': (255, 0, 0),
               'yellow': (0, 255, 255), 'board': (255, 255, 0)}

LANDMARKS = ('target_touch_capture', 'motor_onset_capture', 'hand_exit_capture')


def build_state_machine():
    """A StateMachine carrying the real post-hoc params/methods. No video, no gaze data:
    the eye handler is stubbed (only gaze_sampling_rate is read by store_results, and it is
    restored per participant from the source PKL)."""
    cam = os.path.join(REPO_ROOT, 'calibration/camera_calib.json')
    distortion = DistortionHandler(calibration_json_path=cam, frame_width=1280, frame_height=720)
    board = BoardHandler(aruco_board_cfg_path=os.path.join(REPO_ROOT, 'cfg/game_aruco_board.yaml'),
                         game_cfg_path=os.path.join(REPO_ROOT, 'cfg/game_config.yaml'),
                         colors_dict=colors_dict, colors_list=colors_list, distortion_handler=distortion)
    panel = PanelHandler(panel_configuration_path=os.path.join(REPO_ROOT, 'cfg/sample_shape_cfg'),
                         colors_dict=colors_dict, colors_list=colors_list, distortion_handler=distortion,
                         enable_visualization=False)
    return StateMachine(board, panel, None, distortion,
                        sequence_cfg_path=os.path.join(REPO_ROOT, 'cfg/default_trials_config.yaml'), video_fps=30)


def reset_to_live(m):
    """Undo any previous post-hoc pass so _posthocBump re-derives from the LIVE (causal) marks,
    making the re-processing idempotent regardless of how many times it has been applied."""
    if 'target_touch_live' in m:
        m['target_touch_capture'] = m.pop('target_touch_live')
    if 'motor_onset_live' in m:
        m['motor_onset_capture'] = m.pop('motor_onset_live')
    m.pop('motor_onset_source', None)
    if 'hand_exit_live' in m:
        m['hand_exit_capture'] = m.pop('hand_exit_live')
    if 'hand_exit_live_source' in m:
        m['hand_exit_source'] = m.pop('hand_exit_live_source')
    m.pop('bump', None)


def reprocess(pid, input_root, output_root, topic, dry_run):
    in_dir = os.path.join(input_root, topic, pid)
    pkl = os.path.join(in_dir, f'data_{pid}.pkl')
    if not os.path.exists(pkl):
        log(f"[reprocess::{pid}] no data_{pid}.pkl in {in_dir} -- skipped")
        return None

    sm = build_state_machine()
    with open(pkl, 'rb') as f:
        gsr = pickle.load(f).get('gaze_sampling_rate')
    sm.eye_data_handler = types.SimpleNamespace(gaze_sampling_rate=gsr)
    sm.load_from_pickle(in_dir, pid)

    changed = {k: 0 for k in LANDMARKS}
    n_trials = n_bump = 0
    for key, trial in sm.board_metrics_store.items():
        if key == 'latest':
            continue
        for name, m in trial.items():
            if not isinstance(m, dict) or 'signal_trace' not in m:
                continue                                   # missing / non-real trial
            before = {k: m.get(k) for k in LANDMARKS}
            reset_to_live(m)
            if 'target_touch_capture' in m:                # confirmed touch -> run the bump
                sm._posthocBump(m)
                n_bump += 1
            n_trials += 1
            for k in LANDMARKS:
                if m.get(k) != before[k]:
                    changed[k] += 1

    if not dry_run:
        out_dir = os.path.join(output_root, topic, pid)
        os.makedirs(out_dir, exist_ok=True)
        sm.store_results(output_path=out_dir, participant_id=pid, video_fps=sm.video_fps)

    log(f"[reprocess::{pid}] trials={n_trials} bump={n_bump} | changed vs source: "
        f"touch={changed['target_touch_capture']} motor_onset={changed['motor_onset_capture']} "
        f"hand_exit={changed['hand_exit_capture']}" + ("  (dry-run, not written)" if dry_run else ""))
    return changed


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Re-apply the post-hoc landmark stage from persisted signal traces.")
    ap.add_argument('-p', dest='participant', type=str, default='all', help="participant id, or 'all'")
    ap.add_argument('--input_root', type=str, required=True, help="OutputData_vX root to read PKLs from")
    ap.add_argument('--output_root', type=str, required=True, help="OutputData_vY root to write to (must differ from input)")
    ap.add_argument('--topic', type=str, default='gaze')
    ap.add_argument('--dry_run', action='store_true', help="only report mark changes, do not write")
    ap.add_argument('--report', action='store_true',
                    help="after writing, regenerate the combined CSVs + HTML report by calling "
                         "process_outputs (same aggregator the standard run uses)")
    args = ap.parse_args()

    if os.path.abspath(args.input_root) == os.path.abspath(args.output_root) and not args.dry_run:
        sys.exit("Refusing to overwrite the input: choose a distinct --output_root (or use --dry_run).")

    base = os.path.join(args.input_root, args.topic)
    if args.participant == 'all':
        pids = sorted(d for d in os.listdir(base) if os.path.isdir(os.path.join(base, d)))
    else:
        pids = [args.participant]

    total = {k: 0 for k in LANDMARKS}
    for pid in pids:
        ch = reprocess(pid, args.input_root, args.output_root, args.topic, args.dry_run)
        if ch:
            for k in LANDMARKS:
                total[k] += ch[k]
    log(f"[reprocess] DONE {len(pids)} participant(s) | total changed vs source: "
        f"touch={total['target_touch_capture']} motor_onset={total['motor_onset_capture']} hand_exit={total['hand_exit_capture']}")

    # Optional: regenerate the team artefacts (combined CSVs + HTML report) from the
    # freshly written output, REUSING the standard aggregator -- no duplicated logic.
    if args.report and not args.dry_run:
        from src.tools import process_outputs
        sys.argv = ['process_outputs', '--roots', args.output_root, '-t', args.topic]
        log(f"[reprocess] regenerating combined CSVs + HTML report for {args.output_root} ...")
        process_outputs.main()
