#!/usr/bin/env python3
# encoding: utf-8
import os
import copy

import math
from collections import deque, Counter
import cv2 as cv
import numpy as np

import csv
import pickle
import yaml
from yaml.loader import SafeLoader
from tabulate import tabulate

from src.core.utils import dumpYaml, parseYaml
from src.core.utils import bcolors
from src.core.utils import log, log_debug
from src.core.version import __version__
from src.core.ArucoBoardHandler import detectAllArucos

# Surgical magnifying-glass trace of the touch / hand_exit signals, enabled with
# EEHA_TRACE_TOUCH=1. Dumps one line per watched frame from the REAL code path (run a
# narrow window with --start_frame/--end_frame to debug a single trial faithfully),
# kept separate from the per-gaze-sample --debug_log firehose.
TRACE_TOUCH = bool(os.environ.get('EEHA_TRACE_TOUCH'))

# FIX-1 (v1.2.0, EXPERIMENTAL - OFF by default): advance the target-touch reference
# capture into the panel-removal window so edge / fast-reach targets get a clean
# reference. MEASURED REGRESSION in its current form: the cleanliness gate
# (isTargetAreaClear only rejects the blank panel, NOT the hand) captures a DIRTY
# reference on trials where the hand is already approaching during panel removal, which
# SUPPRESSES the real touch -> net -7 pts touch coverage on a 4-rep A/B (+14 rescued on
# 042/middle rows, -29 lost elsewhere). Needs a stronger cleanliness gate (target cell
# must match its EXPECTED colored content, not skin) before it can be enabled.
EARLY_REF = os.environ.get('EEHA_EARLY_REF', '0') != '0'

def bufferStateChangeMsg(msg):
    return f"{bcolors.BOLD}{bcolors.OKCYAN}{msg}{bcolors.ENDC}\n"

def logErrorMsg(msg):
    log(f"{bcolors.BOLD}{bcolors.ERROR}{msg}{bcolors.ENDC}\n")

def bufferMsg(msg):
    return f"{msg}\n"

def logStateChange(msg):
    log(f"{bcolors.BOLD}{bcolors.OKCYAN}{msg}{bcolors.ENDC}")


def IsSamePanel(panel1, panel2):

    return (panel1['color'] == panel2['color'] and \
           panel1['shape'] == panel2['shape'])


# Default mass threshold for target_found (re-tuned in v1.4.1). SINGLE SOURCE OF TRUTH: the report
# (generate_report) imports this constant, so the report's "objetivo visto" count can never silently
# diverge from the writer's published frame_target_found. Overridable per-run via the env var below.
TARGET_FOUND_MASS_THR_DEFAULT = 0.34


def _phi(z):
    """Standard normal CDF."""
    return 0.5 * (1.0 + math.erf(z / 1.4142135623730951))


# Gauss-Legendre nodes/weights mapped to [0,1], for the bivariate-normal CDF integral below.
_GL_X, _GL_W = np.polynomial.legendre.leggauss(24)
_GL_X = 0.5 * (_GL_X + 1.0)
_GL_W = 0.5 * _GL_W


def _phi2(h, k, rho):
    """Standard bivariate normal CDF P(X<=h, Y<=k) with correlation rho, via Sheppard's
    identity  Phi2(h,k;rho) = Phi(h)Phi(k) + integral_0^rho phi_kernel(t) dt  (Gauss-Legendre).
    Accurate to ~1e-7 vs scipy (validated); lets the gaze ellipse be integrated over a cell
    WITH its tilt, not as two independent axes."""
    if rho > 0.999999:
        rho = 0.999999
    elif rho < -0.999999:
        rho = -0.999999
    base = _phi(h) * _phi(k)
    if abs(rho) < 1e-12:
        return base
    t = _GL_X * rho
    denom = 1.0 - t * t
    kern = np.exp(-(h * h - 2.0 * t * h * k + k * k) / (2.0 * denom)) / (2.0 * np.pi * np.sqrt(denom))
    return base + rho * float(np.dot(_GL_W, kern))


def _cell_mass(mx, my, cov, col, row, ncols, nrows):
    """Gaussian probability mass inside cell (col,row) under N((mx,my), cov), using the FULL
    2x2 covariance (its off-diagonal/tilt, not just the marginal variances). mx,my and the
    cell are in board-normalised coords (0..1). The integral over the axis-aligned cell
    rectangle is Phi2 at its four corners. The earlier marginal (independent-axes) product
    discarded cov[0][1] and erred by up to ~0.16 in cell mass for tilted ellipses -- material
    against target_found_mass_threshold (0.30), so it is replaced here (v1.4.1)."""
    sx = max(cov[0][0], 1e-12) ** 0.5
    sy = max(cov[1][1], 1e-12) ** 0.5
    rho = cov[0][1] / (sx * sy)
    zxl, zxh = (col / ncols - mx) / sx, ((col + 1) / ncols - mx) / sx
    zyl, zyh = (row / nrows - my) / sy, ((row + 1) / nrows - my) / sy
    m = (_phi2(zxh, zyh, rho) - _phi2(zxl, zyh, rho)
         - _phi2(zxh, zyl, rho) + _phi2(zxl, zyl, rho))
    return m if m > 0.0 else 0.0


def _cell_distribution(mx, my, cov, ncols, nrows, topk=3, min_mass=0.02):
    """Top-k cells by Gaussian mass + total on-board mass, from a board-norm centroid and
    full 2x2 covariance. Returns (dist_string, onboard_mass) where dist_string is
    'col,row:mass|...'. ('', '') when no covariance is available."""
    if cov is None or mx is None:
        return '', ''
    cells, onboard = [], 0.0
    for c in range(ncols):
        for r in range(nrows):
            m = _cell_mass(mx, my, cov, c, r, ncols, nrows)
            onboard += m
            if m >= min_mass:
                cells.append((m, c, r))
    cells.sort(reverse=True)
    dist = '|'.join(f"{c},{r}:{m:.2f}" for m, c, r in cells[:topk])
    # disjoint cells -> the integral sums to <=1 already; clamp only against numeric drift.
    return dist, round(min(onboard, 1.0), 3)


class ExceptionNoMoreBlocks(Exception):
    def __init__(self, message):
        super().__init__(message)

"""
    State Machine that handles program execution
"""
class StateMachine:
    def __init__(self, board_handler, panel_handler, eye_data_handler, distortion_handler, sequence_cfg_path, video_fps, slow_analysis = False):
        # Estado init
        self.current_state = "init"
        self.video_fps = video_fps
        # Execution mode is recorded in the output (provenance): the fast path subsamples
        # init/get_test_name and can MISS a marginally-detected panel (trial start), so the
        # result quality must never be silently mode-dependent -- store_results warns if a
        # run came out incomplete, and this flag tells which mode produced the data.
        self.slow_analysis = slow_analysis

        self.board_handler = board_handler
        self.panel_handler = panel_handler
        self.eye_data_handler = eye_data_handler
        self.distortion_handler = distortion_handler

        # Whitelist of all configured marker ids (board + every panel). Detections
        # with any other id are spurious (the detector sometimes fires id 0 on board
        # pieces); they are dropped before any processing or drawing.
        self.valid_aruco_ids = set(self.board_handler.aruco_board_handler.config_ids)
        # ids that belong to a STIMULUS PANEL (not the board): used to flag, per frame, whether
        # a panel is still physically present while the touch is being measured -- a fast touch
        # with the panel not fully removed could contaminate the target occlusion (fT).
        self.panel_aruco_ids = set()
        for panel in self.panel_handler.panel_handler_list:
            self.panel_aruco_ids |= set(panel.config_ids)
            self.valid_aruco_ids |= panel.config_ids

        # Per-marker corner history to damp the homography jitter (see smoothArucos)
        self.aruco_corner_history = {}

        self.undistorted_image = None
        # Last frame in which the board contour was actually detected (not propagated
        # by inertia), used to backdate end_capture to the real occlusion start
        self.last_raw_contour_frame = None


        with open(sequence_cfg_path) as file:
            data = yaml.load(file, Loader=SafeLoader)
            self.test_block_sequence = data['test_block_list']
            self.test_block_count = 0
            self.test_trial_count = 0
            self.trial_id = None
            self.block_id = None

        self.board_metrics_store = {}
        self.board_metrics_now = {}
        self.current_test_key = None

        # Full trace of state-machine transitions: every time the active state changes
        # we log the exact World frame and the block/trial it is associated with, so the
        # analysis team can reconstruct, per participant, when each stage started/ended.
        self.state_transitions = []

        self.norm_coord_list = []
        self.desnormalized_coord_list = []

        self.board_contour_switch_state_threshold = 4
        self.board_contour_nondetected_counter = 0
        self.board_contour_detected_counter = 0
        # ~0.2s of solid visibility required: brief unstable windows (participant
        # still approaching) fired premature starts. init_capture is backdated to the
        # streak start so genuine trials lose no duration
        self.board_contour_start_confirm_threshold = 6
        self.contour_streak_start_frame = None
        self.panel_detected_counter = []
        # Consecutive frames a panel must be seen before it latches. Raised 2 -> 4 to reject
        # spurious / misread ArUco false positives: a misread is a brief blip of ONE marker
        # (e.g. 204 read 2 frames while another card was being removed, faking a red_triangle
        # panel and cascading 3 trials to error in one participant), whereas a real card is
        # shown for many frames. The 2-frame blip no longer reaches the threshold; real panels
        # (incl. single-marker blue_circle/yellow_circle, which persist) are unaffected
        # (validated: 001 misread fixed, 0 real panels lost across 5 participants). A colour
        # check on the card figure was tried first but the fixed/sampled hue did not generalise
        # across participants (white balance) and rejected valid panels -- the temporal
        # persistence is colour-free and robust.
        self.panel_detected_threshold = 4

        ## Target-touch MARK (best-effort, does NOT close the trial): records when the
        # hand TOUCHES the target piece (it is not removed from the board). The touch is
        # a subtle, partial occlusion, so thresholds are moderate; the double-margin
        # requirement keeps false positives down.
        self.target_occlusion_threshold = 0.20   # fraction of changed px in target area
        self.target_occlusion_separation = 0.10  # margin over the control (global) change
        # Per-COLOUR touch threshold (v1.2, data-driven from the 20-participant fT
        # distributions). Real touches peak >=0.22 for every colour, but WARM pieces
        # (red/yellow ~ skin hue) produce a genuinely smaller signal far more often (a
        # warm hand over a warm piece barely changes the colour), so many real touches
        # land at ~0.13-0.20 and were missed (red: 68 misses vs blue 11). Lower thresholds
        # for those; blue/green keep 0.20 (already 88-89%). The control separation guards
        # against false positives; GATE 1 (colour) is also skipped for warm targets.
        self.touch_threshold_by_color = {'red': 0.13, 'yellow': 0.15, 'blue': 0.20, 'green': 0.20}
        self.touch_threshold = self.target_occlusion_threshold
        self.target_is_warm = False
        # Sustained-occlusion frames to confirm a touch. Most trials are short (median
        # ~0.7s) and the touch is a brief gesture, so a long window (6 frames) almost
        # never completed before the trial ended at the border crossing: the touch is
        # watched in test_execution AND test_motor_recovery (see _trackTargetTouch), and
        # 3 frames (~0.1s) is enough to reject a single-frame artefact while catching
        # brief touches.
        self.target_occlusion_confirm_threshold = 2
        self.target_occlusion_counter = 0
        self.target_occlusion_start_frame = None
        # The touch watch does NOT start right after the board appears: the sample
        # panel cardboard is still being removed and sweeps over the target area,
        # which the change detector would read as an occlusion (false touch, and the
        # reference would be captured dirty). It starts only once the target cell is
        # no longer covered by a blank surface (panel gone) AND a short minimum has
        # passed. Detecting the panel actively is more robust than a fixed warmup
        # (the cardboard takes a variable time to be removed). Gaze is still counted
        # from init; only the touch watch is delayed. The warmup is short because
        # isTargetAreaClear is the real guard against the panel sweep; a long warmup
        # ate the whole window on short trials.
        self.target_warmup_frames = 3
        self.target_warmup_end = None
        self.target_tracking_active = False

        # target_found: the eyes must DWELL on the target cell (a sustained
        # fixation), not just sweep over it once. The old rule (first single gaze
        # sample on the target) fired on a fleeting pass on the way elsewhere
        # (false positive). We detect fixations with a dispersion criterion (I-DT):
        # a fixation is a run of >= target_found_min_fixation_samples gaze samples
        # whose board-normalized positions stay within a small spatial spread
        # (bounding-box extent < target_found_fixation_dispersion). A WINDOWED
        # dispersion is robust where a single per-sample velocity is not: two near
        # samples can still belong to a fast transit (the window spreads out), and
        # slow wandering AROUND the target stays compact (still a fixation on it).
        # target_found = the first fixation whose mean per-sample ELLIPSE MASS on the
        # target cell reaches target_found_mass_threshold, marked at the fixation start.
        # This is uncertainty-aware: a fixation hugging the cell boundary (centroid
        # technically inside but the ellipse split with the neighbour) is not forced to
        # a hard yes by a discrete majority vote -- it is found only if enough of its
        # probability mass actually lands on the target. Falls back to the discrete
        # majority vote when a fixation has no covariance (no uncertainty model).
        self.target_found_min_fixation_samples = 6
        # bounding-box extent (dx+dy, board-normalized) below which a window is a
        # fixation. A cell is ~1/8 wide; allow some wander around it. Tunable;
        # validate against measured fixation/transit dispersions on real trials.
        self.target_found_fixation_dispersion = 0.06
        # mean ellipse mass on the target cell (over a fixation) required to call it found.
        # RE-TUNED in v1.4.1 (0.30 -> 0.34) after the bivariate cell-mass fix. Reference point:
        # the geometric 50/50 border (gaze centroid exactly on the target/neighbour edge) is
        # ~0.435 of mass under the correct bivariate integral (measured on the 22-participant
        # v1.4.1 cohort, 12k fixations) -- the old marginal estimator mis-put it at ~0.34. The
        # threshold is set BELOW that border ON PURPOSE: the uncertainty ellipse already absorbs
        # the device error, so a fixation the tracker nudged slightly onto the neighbour but whose
        # ellipse still lands substantial mass on the target should count as "looked at the
        # target". 0.34 rescues those (and stays well above the ~0.2 noise floor) while no longer
        # being as lenient as the old 0.30, which counted fixations whose centre fell clearly
        # outside the cell. Cohort: 1064 of 1283 found at 0.34 (vs 1083 at 0.30, 1029 at 0.38).
        # Overridable via EEHA_TARGET_FOUND_MASS_THR to sweep / re-tune with reprocess_landmarks.
        self.target_found_mass_threshold = float(os.environ.get('EEHA_TARGET_FOUND_MASS_THR', TARGET_FOUND_MASS_THR_DEFAULT))

        ## Motor recovery: after the trial end is decided, keep watching until the
        # board contour comes back in a sustained way (the hand left the board) to
        # record hand_exit, then close. Confirms the touch was final (hand withdrew)
        # and gives the withdrawal time.
        self.motor_recovery_max_frames = 75   # keep watching touch+hand_exit (~2.5s); the
        # touch can peak up to ~57 frames after the border crossing (measured)
        self.motor_recovery_confirm = 3       # sustained contour = hand out of board
        self.motor_recovery_miss_tolerance = 2  # contour flickers as the hand withdraws;
        # a couple of dropped frames must not reset the streak (that lost hand_exit)
        # If the earliest occlusion rise comes more than this many frames after the live
        # contour-loss, that contour-loss had no hand over the board (homography flicker /
        # edge hand not occluding the cells) -> artifact; motor_onset is moved to the real
        # occlusion rise. Below it, the contour-loss is a validated entry and is kept.
        self.motor_onset_artifact_gap = 15
        # A cell must be occluded at least this much (fraction of its area in the board_occ
        # mask) at the press to count as TOUCHED when the target itself was never touched.
        self.wrong_touch_min_occ = 0.15
        # A touched cell must also be FOCAL: occluded clearly more than its cleanest neighbour
        # (occ * focality above this). Rejects the wide uniform arm band, keeps the fingertip.
        self.wrong_touch_min_focality = 0.20
        # Minimum ACCUMULATED focal occlusion for a cell to count as touched (sustained press,
        # not a swept arm cell). Tuned visually on the debug figures.
        self.wrong_touch_min_score = 1.0
        self.motor_recovery_deadline = None
        self.motor_recovery_streak = 0
        self.motor_recovery_misses = 0
        self.motor_recovery_exit_frame = None
        self.pending_finish = None
        self.last_occlusion_measure = None

        ## hand_exit via whole-board occlusion baseline (v1.2.0, decoupled from the
        # touch). The board occlusion is sampled from test_execution (clean reference)
        # and its sustained return to baseline in the motor phase marks the hand
        # leaving, regardless of whether the touch was detected. Falls back to the
        # contour-based path when the board pose is unavailable.
        self.board_occ_active = False       # hand_occ tracking runs (clean ref captured)
        self.board_occ_peak = 0.0
        self.cell_occ_at_peak = None   # per-cell occlusion snapshot at the board_occ peak (touched cell)
        self.cell_occ_at_peak_frame = None
        self.cell_touch_score = None   # accumulated per-cell FOCAL occlusion (sustained fingertip)
        self.board_occ_exit_streak = 0
        self.board_occ_exit_start_frame = None
        self.last_board_occ = None          # last whole-board occlusion (for the trace)
        ## hand_exit via LOCAL target occlusion (fT) returning to baseline. More
        # sensitive than the whole-board occlusion for small reaches (a finger over a
        # single cell barely moves the whole-board fraction but clearly the cell's).
        self.ft_peak = 0.0
        self.ft_exit_streak = 0
        self.ft_exit_start_frame = None
        self.ft_enter_level = 0.20          # fT must have clearly risen (a reach)
        self.ft_exit_level = 0.05           # fT back to ~baseline = hand off the target
        self.ft_exit_confirm = 3
        self.board_occ_enter_level = 0.12   # peak occlusion proving the hand entered
        self.board_occ_exit_level = 0.05    # occlusion back to ~baseline = hand left
        self.board_occ_exit_confirm = 3     # sustained frames to confirm the return
        # POST-HOC hand_exit from the occlusion CURVE (v1.3): the live thresholds miss
        # heavy/slow withdrawals where fT dips after the touch but the NEXT trial's panel
        # re-occludes before the absolute floor is confirmed (measured on the 241 misses).
        # With the full recorded signal_trace we find the withdrawal as the first frame
        # after the touch where fT falls to this FRACTION of the touch peak. Only FILLS
        # misses (never overrides a live hand_exit).
        self.posthoc_exit_ratio = 0.30

        ## DIAGNOSTICS (Fase 0, v1.2.0). Per-trial in-memory time series of the cheap
        # occlusion measures already computed during the watched window. Kept in memory
        # only (not serialised) and consumed at trial close to (a) derive WHY a touch was
        # missed (touch_diag taxonomy) and (b) feed the post-hoc re-threshold fallback,
        # both with zero extra decode. Each entry: (frame, frac_target, frac_control,
        # has_grid, tracking_active). frac_* are None when not measurable that frame.
        self.occlusion_series = []

        ## Per-frame gaze classification, for the debug view: list of
        # (undistorted_coord, kind) with kind in execution/pre_start/on_panel/blank/not_board
        self.gaze_classification = []

        self.tm = cv.TickMeter()

        # Speed up those part that do not need so much precision in processing
        self.frame_speed_multiplier = 1

        self.state_info = {
            "init": {'callback': self.init_state, 'frame_mult': 1 if slow_analysis else int(self.video_fps*0.25)},
            "get_test_name": {'callback': self.get_test_name_state, 'frame_mult': 1 if slow_analysis else int(self.video_fps*0.25)},
            "test_start_execution": {'callback': self.test_start_execution_state, 'frame_mult': 1},
            "test_execution": {'callback': self.test_execution_state, 'frame_mult': 1},
            "test_motor_recovery": {'callback': self.test_motor_recovery_state, 'frame_mult': 1},
            "test_finish_execution": {'callback': self.test_finish_execution_state, 'frame_mult': 1}
        }

        frame_mult = [f"{state_key}: {state_data['frame_mult']} frame skip" for state_key, state_data in self.state_info.items()]
        log("[StateMachine::__init__] Frame multiplication for each state:\n\t· " + '\n\t· '.join(frame_mult))

        state_keys = list(self.state_info.keys())
        # Storage for fixation analysis on each state
        self.fixation_data_store = dict.fromkeys(state_keys+['total'], 0)
        # Storage for frame analysis on each state
        self.frame_data_store = dict.fromkeys(state_keys+['total'], 0)


        self.init_frame_number = math.inf
        self.last_frame_number = math.inf
    
    def visualization(self, original_image, capture_idx, last_capture_idx, frame_width, frame_height, participan_id = ""):

        display_image = self.undistorted_image if self.undistorted_image is not None else original_image
        board_view_cfg, _ = self.board_handler.getVisualization(display_image)
        _, canvas = self.board_handler.getUndistortedVisualization(display_image, self.corners, self.ids)
        if canvas is display_image:
            canvas = display_image.copy()

        ## Sample panel polygon (drawn whenever a panel is detected, e.g. while it
        # is being removed and gaze over it is being discarded)
        panel_polygon = self.panel_handler.getPanelPolygon()
        if panel_polygon is not None:
            cv.polylines(canvas, [panel_polygon.astype(np.int32)], isClosed=True, color=(0,255,255), thickness=2)

        ## Gaze samples over the camera view, drawn with a UNIFIED marker (dark halo +
        # coloured core + thin light ring) so they read clearly on any background and are
        # never confused with the ArUco corner dots. Colour encodes how the sample was used:
        # green=counted (execution), orange=pre_start (counted), magenta=discarded over the
        # panel, gray=blank (panel covering), blue=not_board (off board), cyan=not processed.
        GAZE_COLORS = {'execution': (0,255,0), 'pre_start': (0,165,255), 'on_panel': (255,0,255),
                       'blank': (170,170,170), 'not_board': (255,0,0), 'unprocessed': (255,200,0)}
        if self.norm_coord_list and self.board_handler.display_fixation:
            classification = self.gaze_classification
            if not classification:
                # States that do not project gaze (init/get_test_name): show it anyway
                classification = [(self.distortion_handler.correctCoordinates(coord, homography=None)[0], 'unprocessed')
                                  for coord in self.desnormalized_coord_list]
            for index, (und_coord, kind) in enumerate(classification):
                if und_coord[0] < 0 or und_coord[1] < 0:
                    continue
                cx, cy = int(und_coord[0]), int(und_coord[1])
                color = GAZE_COLORS.get(kind, (255, 255, 255))
                # uncertainty ellipse (image space, 1-sigma + 2-sigma), consistent with the
                # trajectory figures. Skipped for panel / covered-cell gaze (not board gaze).
                cov = self.gaze_cov_list[index] if index < len(self.gaze_cov_list) else None
                if cov is not None and kind not in ('on_panel', 'blank'):
                    Wd, Hd = (float(x) for x in getattr(self, 'world_frame_wh', (1280.0, 720.0)))
                    cpx = np.array([[cov[0][0] * Wd * Wd, cov[0][1] * Wd * Hd],
                                    [cov[1][0] * Hd * Wd, cov[1][1] * Hd * Hd]])
                    wv, vv = np.linalg.eigh(cpx)
                    ang = float(np.degrees(np.arctan2(vv[1, 1], vv[0, 1])))
                    for ksig, th in ((2.0, 1), (1.0, 2)):
                        ax_len = (int(ksig * np.sqrt(max(wv[1], 0.0))), int(ksig * np.sqrt(max(wv[0], 0.0))))
                        if ax_len[0] > 0 and ax_len[1] > 0:
                            cv.ellipse(canvas, (cx, cy), ax_len, ang, 0, 360, color, th, lineType=cv.LINE_AA)
                # centre as an X (matches the trajectory marker)
                cv.drawMarker(canvas, (cx, cy), (20, 20, 20), cv.MARKER_TILTED_CROSS, 16, 4)
                cv.drawMarker(canvas, (cx, cy), color, cv.MARKER_TILTED_CROSS, 13, 2)

        ## Status panel (top-left). proc FPS = processing speed of the software (not
        # the video nor playback rate)
        STATE_LABEL = {'init': 'waiting panel', 'get_test_name': 'panel shown',
                       'test_start_execution': 'panel removed / board appearing',
                       'test_execution': 'SEARCH (gaze on board)',
                       'test_motor_recovery': 'hand on board / waiting hand exit',
                       'test_finish_execution': 'closing trial'}
        debug_data = [f"Participant: {participan_id}   Frame: {capture_idx}/{last_capture_idx}   proc FPS: {self.tm.getFPS():.1f}",
                      f"State: {self.current_state}  ({STATE_LABEL.get(self.current_state, '')})"]
        if self.current_test_key is not None:
            debug_data.append(f"Current Test: Block: {self.test_block_count}, trial: {self.test_trial_count-1};  trial_id: {self.trial_id}")
            debug_data.append(f"   Search -> {self.current_test_key['color']} {self.current_test_key['shape']} (arucos: {self.current_test_key['arucos']})")
            # Event marks detected so far in this trial
            marks = []
            for key, lab in [('init_capture', 'board'), ('motor_onset_capture', 'hand-in')]:
                if key in self.board_metrics_now:
                    marks.append(f"{lab}@{self.board_metrics_now[key]}")
            if self.target_occlusion_counter > 0:
                marks.append(f"target-occl x{self.target_occlusion_counter}")
            if marks:
                debug_data.append("   Marks: " + "  ".join(marks))
        elif 'latest' in self.board_metrics_store and self.board_metrics_store['latest']:
            board_metrics_prev = list(self.board_metrics_store['latest'].values())[0]
            board_metrics_prev_test = list(self.board_metrics_store['latest'].keys())[0]
            debug_data.append(f"Previous Test: trial_id: {board_metrics_prev.get('trial_id', '?')}; Search -> {board_metrics_prev_test}")
            if 'init_capture' in board_metrics_prev and 'end_capture' in board_metrics_prev:
                debug_data.append(f"   Frames {board_metrics_prev['init_capture']} to {board_metrics_prev['end_capture']}"
                                  f" ({board_metrics_prev['end_capture']-board_metrics_prev['init_capture']} frames)")

        box_h = 22 * len(debug_data) + 14
        box_w = int(canvas.shape[1] * 0.55)
        roi = canvas[0:box_h, 0:box_w]
        canvas[0:box_h, 0:box_w] = (roi * 0.35).astype(np.uint8)
        for index, text in enumerate(debug_data):
            cv.putText(canvas, text, (10, 26 + 22*index), cv.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,0), 1, cv.LINE_AA)

        ## PiP (bottom-right): warped board with grid and gaze when the grid is
        # valid; panel view while a panel is being shown; nothing otherwise
        pip = None
        if self.board_handler.cell_matrix is not None and self.board_handler.board_view is not None:
            pip = board_view_cfg
            if self.current_test_key is not None:
                target_coord = self.board_handler.getShapeCoord(self.current_test_key['shape'], self.current_test_key['color'])
                if target_coord[0] is not None:
                    cv.circle(pip, (int(target_coord[0]), int(target_coord[1])), radius=14, color=(255,255,0), thickness=3)
            # Target/control occlusion areas of the end-of-trial detector
            if self.board_handler.last_target_roi is not None:
                x0, y0, x1, y1 = self.board_handler.last_target_roi
                cv.rectangle(pip, (x0, y0), (x1, y1), color=(255,255,0), thickness=2)
            for control_roi in self.board_handler.last_control_rois:
                x0, y0, x1, y1 = control_roi
                cv.rectangle(pip, (x0, y0), (x1, y1), color=(128,128,128), thickness=2)
            if self.last_occlusion_measure is not None:
                cv.putText(pip, f"occl target/ctrl: {self.last_occlusion_measure[0]:.2f}/{self.last_occlusion_measure[1]:.2f}",
                           (10, pip.shape[0]-12), cv.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,0), 2, cv.LINE_AA)
        elif self.panel_handler.getCurrentPanel() is not None:
            pip = self.panel_handler.getVisualization(self.corners, self.ids)

        # Full-resolution warped board (with grid + target/control occlusion ROIs) before it
        # is shrunk into the PiP corner, so documentation figures can use it at full quality.
        self.last_pip_view = pip.copy() if pip is not None else None

        if pip is not None and pip.shape[0] > 0 and pip.shape[1] > 0:
            pip_w = int(canvas.shape[1] * 0.38)
            pip_h = max(1, int(pip.shape[0] * pip_w / pip.shape[1]))
            pip_resized = cv.resize(pip, (pip_w, pip_h))
            y0 = canvas.shape[0] - pip_h - 8
            x0 = canvas.shape[1] - pip_w - 8
            cv.rectangle(canvas, (x0-2, y0-2), (x0+pip_w+1, y0+pip_h+1), (255,255,255), 2)
            canvas[y0:y0+pip_h, x0:x0+pip_w] = pip_resized

        return canvas, cv.resize(canvas, (frame_width, frame_height))
        
    def processPanel(self, undistorted_image, capture_idx, desnormalized_coord_list):
        # processPanel may be invoked more than once for the SAME video frame (the step()
        # while-loop chains callbacks); evaluate ONCE per frame so the persistence counter
        # measures N consecutive VIDEO frames, not N callback steps.
        if capture_idx == getattr(self, '_panel_eval_frame', None):
            return self._panel_eval_result
        self._panel_eval_frame = capture_idx

        current_panel = None
        self.panel_handler.step(undistorted_image, self.corners, self.ids)
        current_detected_panel = self.panel_handler.getCurrentPanel()

        if current_detected_panel is None:
            self.panel_detected_counter = []
        elif (not self.panel_detected_counter
              or not IsSamePanel(current_detected_panel, self.panel_detected_counter[-1])):
            # first detection OR a DIFFERENT panel -> (re)start the consecutive count on it.
            # Resetting on a different panel is essential: otherwise a new panel B could never
            # reach the threshold while the counter stays latched on the old panel A.
            self.panel_detected_counter = [current_detected_panel]
        else:
            self.panel_detected_counter.append(current_detected_panel)

        if current_detected_panel is not None and len(self.panel_detected_counter) >= self.panel_detected_threshold:
            current_panel = current_detected_panel

        self._panel_eval_result = current_panel
        return current_panel

    ## Averages each marker's corners over the last few frames to damp the sub-pixel
    # detection jitter that, amplified through the homography, made the warped board
    # content "dance" and faked target occlusions. The history is keyed by marker id
    # and dropped for markers not seen this frame (so a reappearing marker does not
    # average against stale positions). A short window follows real head motion with
    # negligible lag while removing the jitter.
    def smoothArucos(self, corners, ids, window=4):
        if ids is None:
            self.aruco_corner_history.clear()
            return corners, ids
        seen = set()
        smoothed = []
        for idx, marker_id in enumerate(np.asarray(ids).flatten()):
            marker_id = int(marker_id)
            seen.add(marker_id)
            hist = self.aruco_corner_history.setdefault(marker_id, deque(maxlen=window))
            hist.append(corners[idx])
            smoothed.append(np.mean(np.array(hist), axis=0).astype(np.float32))
        for marker_id in list(self.aruco_corner_history.keys()):
            if marker_id not in seen:
                del self.aruco_corner_history[marker_id]
        return smoothed, ids

    ## Drops detections whose id is not a configured marker (spurious detections on
    # board pieces). Keeps corners and ids aligned.
    def filterValidArucos(self, corners, ids):
        if ids is None:
            return corners, ids
        flat = np.asarray(ids).flatten()
        keep = [i for i, marker_id in enumerate(flat) if int(marker_id) in self.valid_aruco_ids]
        if len(keep) == len(flat):
            return corners, ids
        if not keep:
            return [], None
        filtered_corners = [corners[i] for i in keep]
        filtered_ids = np.array([[int(flat[i])] for i in keep], dtype=np.int32)
        return filtered_corners, filtered_ids

    def getFrameMultiplier(self):
        return max(1, self.frame_speed_multiplier)

    """
        State machine control loop
    """
    def step(self, original_image, capture_idx):
        self.tm.start()

        ## Undistort the image (alpha=0, full resolution) for the board/panel handlers.
        self.undistorted_image = self.distortion_handler.undistortImage(original_image)
        ## ArUcos are detected on the ORIGINAL (distorted) image, NOT the undistorted one: the
        # undistort pushes the edge markers — especially the top row — out of frame and they
        # are lost (measured: participant 042 lost markers in 16/16 frames, ~5 per frame). The
        # detected corners are then undistorted to the alpha=0 image space (newK == K, so they
        # match the gaze projection to 0.15 px), feeding the homography more markers without
        # any change in resolution or in the gaze coordinates.
        corners_dist, self.ids = detectAllArucos(original_image)
        self.corners = [self.distortion_handler.correctCoordinates(c.reshape(-1, 2), homography=None)
                            .reshape(1, -1, 2).astype(np.float32)
                        for c in corners_dist]
        self.corners, self.ids = self.filterValidArucos(self.corners, self.ids)
        # NOTE: aruco corner smoothing (smoothArucos) was tried to stabilise the
        # homography but it degraded contour detection and lost trials; reverted.

        self.init_frame_number = min(self.init_frame_number, capture_idx)
        self.last_frame_number = capture_idx

        self.norm_coord_list = self.eye_data_handler.step(capture_idx)
        self.gaze_cov_list = getattr(self.eye_data_handler, 'last_cov_list', None) or []
        self.gaze_classification = []
        # Actual world-frame size: gaze means are denormalised against it (below) and the
        # per-sample covariance must be scaled to the SAME pixel space in _projectGazeCov
        # (v1.4.1 -- was hardcoded 1280x720, which silently mis-scaled the covariance on any
        # non-720p video while the mean stayed correct). Warn once if the frame is not the
        # size the per-participant uncertainty MODEL was calibrated at (1280x720).
        self.world_frame_wh = (original_image.shape[1], original_image.shape[0])
        if self.world_frame_wh != (1280, 720) and not getattr(self, '_warned_frame_size', False):
            self._warned_frame_size = True
            log(f"[StateMachine] WARNING: world frame {self.world_frame_wh} != (1280, 720); the gaze "
                f"uncertainty model (GazeUncertaintyModel/EyeDataHandler) assumes 720p px units -- "
                f"covariance magnitude may be mis-scaled. Re-check the calibration px units.")
        self.desnormalized_coord_list = []
        for norm_coord in self.norm_coord_list:
            # Gaze normalized coords are relative to the original (distorted) frame
            desnormalized_x = int(norm_coord[0] * original_image.shape[1])
            desnormalized_y = int(norm_coord[1] * original_image.shape[0])
            self.desnormalized_coord_list.append(np.array([[desnormalized_x, desnormalized_y]]))
            self.fixation_data_store['total'] += 1

        # Propagate this frame to next state to no to lose steps
        previous_state = None
        while self.current_state != previous_state:
            previous_state = self.current_state
            self.state_info[self.current_state]['callback'](self.undistorted_image, capture_idx, self.desnormalized_coord_list)
            self.frame_speed_multiplier = self.state_info[self.current_state]['frame_mult']
            # A callback may switch the state (possibly several times within this frame):
            # record each transition with its exact frame and the block/trial it belongs to.
            if self.current_state != previous_state:
                self._recordTransition(previous_state, self.current_state, capture_idx)
        
        if self.norm_coord_list: self.fixation_data_store[self.current_state] += len(self.norm_coord_list)
        self.frame_data_store[self.current_state] += 1
        self.frame_data_store['total'] += 1
        self.board_metrics_now['status'] = self.current_state
        
        # key = cv.waitKey()
        # if key == ord('q') or key == ord('Q') or key == 27:
        #     exit()
        
        self.tm.stop()

    def _recordTransition(self, from_state, to_state, capture_idx):
        # block/trial reflect the trial active right after the callback (init advances
        # them when it accepts a panel; on close they still point at the trial that ended)
        trial_name = ''
        if self.current_test_key is not None:
            trial_name = f"{self.current_test_key['color']}_{self.current_test_key['shape']}"
        self.state_transitions.append({'frame': capture_idx,
                                       'block': self.block_id,
                                       'trial': self.trial_id,
                                       'trial_name': trial_name,
                                       'from_state': from_state,
                                       'to_state': to_state})

    ## Used in case something happens and some state change is missed: if a DIFFERENT
    # panel appears while a trial is being set up or run, the current trial is closed
    # and we resync (see _handleUnexpectedPanel). Returns True if it fired.
    def is_error_init_state(self, undistorted_image, capture_idx, desnormalized_coord_list):
        current_panel = self.processPanel(undistorted_image, capture_idx, desnormalized_coord_list)
        if current_panel is not None and not IsSamePanel(current_panel, self.current_test_key):
            return self._handleUnexpectedPanel(current_panel, capture_idx)
        return False

    ## Single place that handles a state JUMP caused by a different panel appearing
    # mid-trial. Used from test_start_execution / test_execution (via is_error_init_state)
    # AND get_test_name, so the jump is closed the same way everywhere. With gathered data
    # the trial is closed as a valid by_next_panel (end backdated to the last board
    # sighting); without data it is a transition_error_no_init. Then we go back to init so
    # the new panel is matched against the expected sequence. Returns True.
    def _handleUnexpectedPanel(self, current_panel, capture_idx):
        if 'init_capture' in self.board_metrics_now:
            # Trial was running: the hand occlusion lost the race against the next
            # panel detection. Close it at the last frame the board was visible.
            end_capture = self.last_raw_contour_frame if self.last_raw_contour_frame is not None else capture_idx
            self.board_metrics_now['end_capture'] = end_capture
            self.trimTrialToFrame(self.board_metrics_now, end_capture)
            self.board_metrics_now['status'] = 'test_finish_by_next_panel'
            self.board_metrics_now['trial_id'] = self.trial_id
            self._deriveTouchDiagnostics(self.board_metrics_now)
            key = f"{self.current_test_key['color']}_{self.current_test_key['shape']}"
            logStateChange(f"[StateMachine::handleUnexpectedPanel] [{capture_idx}] New panel ({current_panel}) while trial for {self.current_test_key} was running. Closed at frame {end_capture} as test_finish_by_next_panel.")
        else:
            self.board_metrics_now['transition_error'] = {'transition_error': {True: 0, False: 0}}
            self.board_metrics_now['end_capture'] = capture_idx
            self.board_metrics_now['status'] = self.current_state
            self.board_metrics_now['trial_id'] = self.trial_id
            key = f"transition_error_no_init_{self.current_test_key['color']}_{self.current_test_key['shape']}"
            logErrorMsg(f"[StateMachine::handleUnexpectedPanel] [{capture_idx}] New panel ({current_panel}) appeared before {self.current_test_key} gathered any data. transition_error_no_init.")

        self.board_metrics_store[(self.block_id, self.trial_id)] = {key: copy.deepcopy(self.board_metrics_now)}
        self.board_metrics_store['latest'] = self.board_metrics_store[(self.block_id, self.trial_id)]
        self.resetTrialTracking()

        self.current_state = "init"
        return True

    ## Whether the detected panel still appears somewhere in the not-yet-consumed part
    # of the expected sequence. Used to reject a spurious / out-of-sequence panel in init
    # instead of consuming the whole remaining sequence as missing trials (which
    # terminated the run). Read-only: does not touch the sequence counters.
    def _detectedInRemaining(self, detected_trial):
        block_count = self.test_block_count
        trial_count = self.test_trial_count
        while block_count < len(self.test_block_sequence):
            block = self.test_block_sequence[block_count][1]
            while trial_count < len(block):
                if block[trial_count][1] == detected_trial:
                    return True
                trial_count += 1
            block_count += 1
            trial_count = 0
        return False

    def init_state(self, undistorted_image, capture_idx, desnormalized_coord_list):
        current_panel = self.processPanel(undistorted_image, capture_idx, desnormalized_coord_list)
        if current_panel is not None:
            detected_trial = f"{current_panel['color']}_{current_panel['shape']}"

            # Reject a spurious / out-of-sequence panel: if it does not appear anywhere
            # in the remaining expected sequence, ignore it (stay in init) instead of
            # marking every remaining trial as missing and terminating the run.
            if not self._detectedInRemaining(detected_trial):
                logErrorMsg(f"[StateMachine::init_state] [{capture_idx}] Detected panel {detected_trial} is not in the remaining expected sequence (spurious / out of order); ignored, staying in init.")
                return

            self.current_test_key = current_panel

            ## Check that is the expected configured sequence or add errored tests until current one
            # is found in the sequence:
            found_trial = False
            while self.test_block_count < len(self.test_block_sequence):
                test_block = self.test_block_sequence[self.test_block_count][1]
                self.trial_id = None
                self.block_id = None
                while self.test_trial_count < len(test_block):
                    expected_trial = self.test_block_sequence[self.test_block_count][1][self.test_trial_count][1]
                    self.trial_id = self.test_block_sequence[self.test_block_count][1][self.test_trial_count][0]
                    self.block_id = self.test_block_sequence[self.test_block_count][0]
                    self.test_trial_count += 1 # Next step in Init needs this counter to advance whether the test is correct or not :)
                    if expected_trial != detected_trial:
                        logErrorMsg(f"[StateMachine::init_state] ERROR, expected trial ([{self.block_id},{self.trial_id}]) did not happend. {expected_trial = }; {detected_trial = } (detected arucos: {self.current_test_key['arucos']})")
                        board_metrics_now = {'end_capture': -1, 'init_capture': -1, 'sequence': [],
                                             'missing_trial_error': {'missing_trial_error': {True: 0, False: 0}},
                                             'trial_id': self.trial_id, 'status': None}
                        
                        self.board_metrics_store[(self.block_id, self.trial_id)] = {f"missing_trial_error_{expected_trial}": copy.deepcopy(board_metrics_now)}
                        self.board_metrics_store['latest'] = self.board_metrics_store[(self.block_id, self.trial_id)]
                    else:
                        found_trial = True
                        break

                if found_trial:
                    break
                if self.test_block_count + 1 < len(self.test_block_sequence):
                    if (self.test_trial_count >= len(self.test_block_sequence[self.test_block_count][1])):
                        self.test_block_count += 1
                        self.test_trial_count = 0
                else:
                    logErrorMsg("[StateMachine::init_state] ERROR: No more blocks available in test sequence.")
                    self.handle_end_of_video()
                    raise ExceptionNoMoreBlocks("END StateMachineHandler, no more blocks available.")
                    break

            if found_trial:
                self.current_state = "get_test_name"
                logStateChange(f"[StateMachine::init_state] [{capture_idx}] Switch to get_test_name state. Test panel detected.")

    def get_test_name_state(self, undistorted_image, capture_idx, desnormalized_coord_list):

        current_panel = self.processPanel(undistorted_image, capture_idx, desnormalized_coord_list)
        if current_panel is None:
            self.current_state = "test_start_execution"
            logStateChange(f"[StateMachine::get_test_name] [{capture_idx}] Switch to test_start_execution. Gathering data for test {self.current_test_key['shape']} {self.current_test_key['color']} [Block id:{self.block_id}; trial id:{self.trial_id};] (detected arucos: {self.current_test_key['arucos']})")
        elif not IsSamePanel(current_panel, self.current_test_key):
            # Panel swapped before being removed (a jump): abandon this one (no data yet)
            # and resync through init instead of waiting for it to disappear forever.
            self._handleUnexpectedPanel(current_panel, capture_idx)

    def test_start_execution_state(self, undistorted_image, capture_idx, desnormalized_coord_list):
        if self.is_error_init_state(undistorted_image, capture_idx, desnormalized_coord_list): return

        self.board_handler.step(undistorted_image, self.corners, self.ids,
                                panel_polygon=self.panel_handler.getPanelPolygon())

        # Gaze can already be projected while the panel is being removed (board pose
        # from arucos + session reference grid); stored flagged as pre_start
        self.processEarlyGaze(capture_idx, desnormalized_coord_list)

        # FIX-1: prime the touch reference here, in the same permissive window the early
        # gaze uses, so edge targets get a clean reference before the hand arrives.
        self._primeTouchReference(capture_idx)

        # Require a few consecutive raw detections before starting the trial: a single
        # noisy detection produced degenerate, near-empty trials. init_capture is then
        # backdated to the first frame of the confirmed streak (see test_execution)
        if self.board_handler.contour_detected_raw:
            if self.board_contour_detected_counter == 0:
                self.contour_streak_start_frame = capture_idx
            self.board_contour_detected_counter += 1
        else:
            self.board_contour_detected_counter = 0

        if self.board_contour_detected_counter >= self.board_contour_start_confirm_threshold:
            self.board_contour_detected_counter = 0
            self.current_state = "test_execution"
            logStateChange(f"[StateMachine::test_start_execution] [{capture_idx}] Switch to test_execution. Gathering data for test {self.current_test_key['shape']} {self.current_test_key['color']} (detected arucos: {self.current_test_key['arucos']})")
            

    def test_execution_state(self, undistorted_image, capture_idx, desnormalized_coord_list):
        if not 'init_capture' in self.board_metrics_now:
            # Backdate the start to the first frame of the confirmed detection streak
            init_capture = self.contour_streak_start_frame if self.contour_streak_start_frame is not None else capture_idx
            self.board_metrics_now['init_capture'] = init_capture
            self.contour_streak_start_frame = None
            self.board_metrics_now['trial_id'] = self.trial_id
            self.board_metrics_now['target_cord'] = list(self.board_handler.getShapeCellIndex(self.current_test_key['shape'], self.current_test_key['color']))
            self.board_metrics_now['target_norm_coord'] = self.board_handler.getPixelBoardNorm(
                                                                [self.board_handler.getShapeCoord(self.current_test_key['shape'], self.current_test_key['color'])]
                                                            ).tolist()
            if 'sequence' not in self.board_metrics_now:
                self.board_metrics_now['sequence'] = []
            self.last_raw_contour_frame = capture_idx

            # Pre-start samples inside the confirmed streak belong to the trial
            # itself (init_capture was backdated over them): relabel and count them
            for sample in self.board_metrics_now['sequence']:
                if sample.get('phase') == 'pre_start' and sample['frame'] >= init_capture:
                    sample['phase'] = 'execution'
                    color, shape, slot = sample['color'], sample['shape'], sample['slot']
                    if color not in self.board_metrics_now:
                        self.board_metrics_now[color] = {shape: {True: 0, False: 0}}
                    if shape not in self.board_metrics_now[color]:
                        self.board_metrics_now[color][shape] = {True: 0, False: 0}
                    self.board_metrics_now[color][shape][slot] += 1

            # Start tracking the target area appearance for the end-of-trial detection,
            # but only after a warmup so the panel removal does not fake a touch.
            # Skipped when FIX-1 already primed a CLEAN reference in the panel-removal
            # window (target_tracking_active True): re-initialising here would discard it.
            self.target_is_warm = str(self.current_test_key['color']) in ('red', 'yellow')
            self.touch_threshold = self.touch_threshold_by_color.get(
                str(self.current_test_key['color']), self.target_occlusion_threshold)
            if not self.target_tracking_active:
                self.board_handler.initTargetTracking(self.board_metrics_now['target_cord'],
                                                      target_color=self.current_test_key['color'])
                self.target_occlusion_counter = 0
                self.target_occlusion_start_frame = None
                self.target_warmup_end = capture_idx + self.target_warmup_frames
                self.target_tracking_active = False

            # hand_occ tracking (hand_exit): capture a CLEAN whole-board
            # reference now -- the board just passed the contour-confirm streak, so it is
            # clean -- and run it every frame from here, independent of the touch tracking.
            self.board_handler.board_occ_ref = None
            self.board_occ_active = True
            self.board_occ_peak = 0.0
            self.cell_occ_at_peak = None   # per-cell occlusion snapshot at the board_occ peak (touched cell)
            self.cell_occ_at_peak_frame = None
            self.cell_touch_score = None   # accumulated per-cell FOCAL occlusion (sustained fingertip)

        if self.is_error_init_state(undistorted_image, capture_idx, desnormalized_coord_list): return

        self.board_handler.step(undistorted_image, self.corners, self.ids,
                                panel_polygon=self.panel_handler.getPanelPolygon())
        if self.board_handler.contour_detected_raw:
            self.last_raw_contour_frame = capture_idx
        coord_data_list = self.board_handler.getPixelInfo(desnormalized_coord_list)

        for index, coord_data in enumerate(coord_data_list):
            color, shape, slot, board_coord, corrected_coord = coord_data
            cov_img = self.gaze_cov_list[index] if index < len(self.gaze_cov_list) else None
            self.recordGazeSample(capture_idx, color, shape, slot, board_coord, corrected_coord, phase='execution',
                                  cov_img=cov_img, desnorm_coord=desnormalized_coord_list[index])
            und_coord = self.distortion_handler.correctCoordinates(desnormalized_coord_list[index], homography=None)[0]
            self.gaze_classification.append((und_coord, 'execution' if color != 'not_board' else 'not_board'))

        ## TARGET TOUCH (best-effort MARK, does NOT close the trial). The trial END is
        # decided by the contour loss below (the v1.0 criterion), which keeps the exact
        # v1.0 timing. The touch watch continues into test_motor_recovery (see
        # _trackTargetTouch): the touch physically happens when the hand is already over
        # the board (contour being lost), so watching it only here missed most touches.
        self._trackTargetTouch(capture_idx)
        # Seed the whole-board occlusion reference on the clean board and build its peak
        # as the hand enters, so the motor phase can detect the return (hand_exit).
        self._trackBoardOcclusion(capture_idx)
        self._traceTouch(capture_idx, 'execution')

        ## TRIAL END: sustained board-contour loss (the v1.0 criterion, restores the
        # exact timing so the state machine stays in sync). The hand entering the
        # board is recorded as motor_onset; the end is backdated to the last frame the
        # board was actually seen.
        if not self.board_handler.isContourDetected():
            self.board_contour_nondetected_counter += 1
        else:
            self.board_contour_nondetected_counter = 0

        if self.board_contour_nondetected_counter > self.board_contour_switch_state_threshold:
            self.board_contour_nondetected_counter = 0
            end_capture = self.last_raw_contour_frame if self.last_raw_contour_frame is not None else capture_idx
            if 'motor_onset_capture' not in self.board_metrics_now:
                self.board_metrics_now['motor_onset_capture'] = end_capture
            self.finishTrial(end_capture, 'test_finish_execution', capture_idx, reason='board contour lost')

    ## Best-effort target-touch MARK. Called from test_execution AND test_motor_recovery:
    # the physical touch occurs when the hand is already over the board (the contour is
    # being lost and the trial is ending), so restricting the watch to test_execution
    # missed most touches. The warp (aruco homography) and grid (stable reference rect)
    # survive partial occlusion, so getTargetOcclusionMeasure can still read the target
    # area while the hand reaches it. Records the START of the confirmed occlusion streak
    # (the touch onset). Does not close the trial, so a false/late fire is bounded.
    def _trackTargetTouch(self, capture_idx):
        self.last_occlusion_measure = None
        if 'target_touch_capture' in self.board_metrics_now:
            # Touch already found: keep measuring the target occlusion (for the fT-return
            # hand_exit) but stop re-detecting the touch.
            if self.target_tracking_active:
                self.last_occlusion_measure = self.board_handler.getTargetOcclusionMeasure()
            return
        # Diagnostics: whether the grid (cell_matrix) is available this watched frame.
        # Its absence (too few ArUcos) is the few_arucos failure mode and the only
        # place where the white-grid localization fallback can help (roadmap A.1).
        has_grid = self.board_handler.cell_matrix is not None
        # Activation needs the board clear of the panel; it can only latch while the
        # contour is still seen (in test_execution). Once active it stays active into
        # the motor phase.
        if not self.target_tracking_active:
            # With a session template the reference is clean regardless of the live frame,
            # so we no longer need a clean per-trial window (contour_detected_raw): activate
            # as soon as the panel is gone (target area not blank) and the grid is available.
            # This is what rescues EDGE targets, where the hand covers the target from the
            # start and the contour is already lost. Without a template (very first trial)
            # we keep the old strict gate. isTargetAreaClear still rejects the panel sweep.
            has_template = self.board_handler.session_template is not None
            if capture_idx >= self.target_warmup_end and self.board_handler.cell_matrix is not None \
               and self.board_handler.isTargetAreaClear(self.board_metrics_now.get('target_cord')) \
               and (self.board_handler.contour_detected_raw or has_template):
                self.target_tracking_active = True
        if not self.target_tracking_active:
            self.occlusion_series.append((capture_idx, None, None, has_grid, False))
            return
        self.last_occlusion_measure = self.board_handler.getTargetOcclusionMeasure()
        if self.last_occlusion_measure is not None:
            frac_target, frac_control = self.last_occlusion_measure
            self.occlusion_series.append((capture_idx, float(frac_target), float(frac_control), has_grid, True))
            if frac_target > self.touch_threshold \
               and (frac_target - frac_control) > self.target_occlusion_separation:
                if self.target_occlusion_counter == 0:
                    self.target_occlusion_start_frame = capture_idx
                self.target_occlusion_counter += 1
            else:
                self.target_occlusion_counter = 0
        else:
            self.occlusion_series.append((capture_idx, None, None, has_grid, True))
        if self.target_occlusion_counter >= self.target_occlusion_confirm_threshold:
            self.board_metrics_now['target_touch_capture'] = self.target_occlusion_start_frame

    ## FIX-1: captures the touch reference in the permissive panel-removal window (the
    # same place the early gaze already projects onto the board), decoupled from the
    # board CONTOUR. Edge / fast-reach targets are occluded by the hand by the time the
    # contour-driven test_execution starts, so the strict (contour-gated) activation
    # never finds a clean frame; here we capture it as soon as the board pose + grid are
    # available and the target cell is clear of the panel. Captures ONCE per trial.
    def _primeTouchReference(self, capture_idx):
        if not EARLY_REF or self.target_tracking_active or self.current_test_key is None:
            return
        if self.board_handler.cell_matrix is None or self.board_handler.homography is None:
            return
        target_cell = list(self.board_handler.getShapeCellIndex(
            self.current_test_key['shape'], self.current_test_key['color']))
        if target_cell[0] is None or not self.board_handler.isTargetAreaClear(target_cell):
            return
        self.board_handler.initTargetTracking(target_cell, target_color=self.current_test_key['color'])
        self.board_handler.getTargetOcclusionMeasure()  # first call captures the clean ref
        self.board_handler.getBoardOcclusionMeasure()
        self.target_occlusion_counter = 0
        self.target_occlusion_start_frame = None
        self.target_warmup_end = capture_idx          # panel already gone: no extra warmup
        self.target_tracking_active = True
        if TRACE_TOUCH:
            log(f"[TRACE prime       {capture_idx}] early clean touch reference for cell {target_cell}")

    ## Samples the whole-board occlusion and tracks its peak. Only runs once target
    # tracking is active so the reference getBoardOcclusionMeasure captures the FIRST
    # time is the CLEAN board (the hand has not reached yet), never the hand already
    # over it. Called from test_execution (to seed the clean reference and build the
    # peak as the hand enters) and from test_motor_recovery (to detect the return).
    # Returns the current occlusion fraction, or None when not sampled/observable.
    def _trackBoardOcclusion(self, capture_idx):
        if not self.board_occ_active:
            self.last_board_occ = None
            return None
        board_occ = self.board_handler.getBoardOcclusionMeasure()
        self.last_board_occ = board_occ
        if board_occ is not None:
            # board_occ_peak (the hand was clearly over the board) gates the hand_exit
            # detection below. The contour-based motor_onset is the hand-entry mark.
            # Per-cell occlusion RUNNING MAX over the reach+recovery: captures which cells the
            # hand covered even for a LOCAL touch that barely moves the whole-board board_occ
            # (those never reached a board_occ peak and were missed before). The touched cell is
            # then picked by focality (fingertip), not by the raw max (which is the arm).
            cell_map = self.board_handler.getCellOcclusionMap()
            if cell_map is not None:
                self.cell_occ_at_peak = cell_map if self.cell_occ_at_peak is None \
                    else np.maximum(self.cell_occ_at_peak, cell_map)
                # Accumulate per-cell FOCALITY over the whole reach. A fingertip RESTS on one
                # cell (focal every frame -> high accumulated score); the arm SWEEPS (each cell
                # focal only in passing -> low score). The touched cell = the sustained-focal max.
                foc = self._focalOcc(cell_map)
                self.cell_touch_score = foc if self.cell_touch_score is None else self.cell_touch_score + foc
            if board_occ > self.board_occ_peak:
                self.cell_occ_at_peak_frame = capture_idx   # ~ the press (max hand over board)
            self.board_occ_peak = max(self.board_occ_peak, board_occ)

    def _focalOcc(self, occ):
        """Per-cell focality = how much MORE occluded a cell is than its surroundings (median of
        its 8 neighbours), clamped >= 0 and only where the cell itself is occluded. A fingertip
        pressing one cell is focal; the wide uniform arm band is not (neighbours equally covered)."""
        occ = np.asarray(occ, dtype=float)
        R, C = occ.shape
        foc = np.zeros_like(occ)
        for r in range(R):
            for c in range(C):
                if occ[r, c] < self.wrong_touch_min_occ:
                    continue
                neigh = [occ[r + dr, c + dc] for dr in (-1, 0, 1) for dc in (-1, 0, 1)
                         if (dr or dc) and 0 <= r + dr < R and 0 <= c + dc < C]
                if neigh:
                    foc[r, c] = max(0.0, occ[r, c] - float(np.median(neigh)))
        return foc

    ## Surgical per-frame trace of the touch / hand_exit signals from the REAL code
    # (EEHA_TRACE_TOUCH=1). Run a narrow window with --start_frame/--end_frame.
    def _traceTouch(self, capture_idx, phase):
        occ = self.last_occlusion_measure
        grid = 'ref' if getattr(self.board_handler, 'grid_from_reference', False) \
               else ('cont' if self.board_handler.cell_matrix is not None else 'none')
        # PERSIST a compact per-frame signal trace (always on, cheap: one small dict
        # per watched frame). It records the touch/hand signals over time so the entry
        # and exit can be detected/validated POST-HOC with the FULL occlusion profile
        # (lookahead): a border cut is a real hand only if the occlusion then rises; a
        # hand_exit only if it then stays low. Also enables flagging border false
        # positives (contour lost while the board pose was unreliable -> grid='ref').
        # Panel still present this frame: any stimulus-panel ArUco among the detections means
        # the card has not been fully removed -- a touch measured while panel=1 may have its
        # target occlusion (fT) contaminated by the panel, not the finger (flaggable post-hoc).
        panel_now = int(bool(self.panel_aruco_ids & set(np.asarray(self.ids).flatten().tolist()))) \
            if self.ids is not None else 0
        self.board_metrics_now.setdefault('signal_trace', []).append({
            'f': capture_idx, 'ph': phase[:4],
            'fT': round(occ[0], 3) if occ is not None else None,     # target occlusion
            'fC': round(occ[1], 3) if occ is not None else None,     # control occlusion (separation)
            'bocc': round(self.last_board_occ, 3) if self.last_board_occ is not None else None,
            'cont': int(self.board_handler.contour_detected_raw),
            'homog': int(self.board_handler.homography is not None),
            'panel': panel_now,
            'grid': grid, 'act': int(self.target_tracking_active)})
        if not TRACE_TOUCH:
            return
        fT = f"{occ[0]:.2f}" if occ is not None else "  - "
        fC = f"{occ[1]:.2f}" if occ is not None else "  - "
        bo = f"{self.last_board_occ:.3f}" if self.last_board_occ is not None else "  -  "
        log(f"[TRACE {phase:11s} {capture_idx}] act={int(self.target_tracking_active)} "
            f"H={int(self.board_handler.homography is not None)} cont={int(self.board_handler.contour_detected_raw)} "
            f"grid={grid:>4s} fT={fT} fC={fC} tcnt={self.target_occlusion_counter} "
            f"bocc={bo} bpeak={self.board_occ_peak:.3f} bexit={self.board_occ_exit_streak} "
            f"touch={'Y' if 'target_touch_capture' in self.board_metrics_now else '.'} "
            f"hexit={'Y' if 'hand_exit_capture' in self.board_metrics_now else '.'}")

    ## Keys of board_metrics that are not color counters
    METADATA_KEYS = ('init_capture', 'end_capture', 'early_init_capture', 'motor_onset_capture',
                     'target_touch_capture', 'hand_exit_capture', 'sequence', 'trial_id', 'status',
                     'target_cord', 'target_norm_coord', 'transition_error', 'missing_trial_error',
                     'touch_diag', 'hand_exit_source', 'signal_trace',
                     'hand_exit_live', 'hand_exit_live_source', 'bump',
                     'target_touch_live', 'motor_onset_live', 'motor_onset_source', 'cell_occ_peak',
                     'cell_occ_peak_frame', 'cell_touch_score', 'touched_cell', 'touched_piece', 'gaze_validated_cell',
                     'gaze_validated_piece', 'frame_validation', 'wrong_touch_frame', 'error_type')

    ## Derives, from the in-memory occlusion series, WHY the target touch was (not)
    # detected, so the fallbacks can be targeted and the misses measured. Pure
    # post-processing of cheap scalars already gathered this trial; stored as a small
    # dict in board_metrics (not the full series). Reasons:
    #   confirmed        - touch was detected
    #   few_arucos       - grid (cell_matrix) missing part of the window: no homography
    #                      enough to place the ROI -> white-grid fallback territory (A.1)
    #   never_activated  - tracking never latched (panel sweep / contour never clean)
    #   fT_below         - signal stayed below the occlusion threshold (reach geometry:
    #                      the hand did not occlude the target ROI, A.2)
    #   control_ge       - target crossed the threshold but never separated from control
    #                      (arm occludes control cells as much as the target, A.3)
    #   unconfirmed      - crossed both but not for enough consecutive frames
    def _deriveTouchDiagnostics(self, metrics):
        series = self.occlusion_series
        confirmed = 'target_touch_capture' in metrics
        active = [s for s in series if s[4] and s[1] is not None]
        n_active = len(active)
        frac_grid_missing = (sum(1 for s in series if not s[3]) / len(series)) if series else 0.0
        max_fT = max((s[1] for s in active), default=0.0)
        max_sep = max((s[1] - s[2] for s in active), default=0.0)
        thr = self.touch_threshold
        if confirmed:
            reason = 'confirmed'
        elif n_active == 0:
            reason = 'few_arucos' if frac_grid_missing > 0.5 else 'never_activated'
        elif max_fT <= thr:
            reason = 'fT_below'
        elif max_sep <= self.target_occlusion_separation:
            reason = 'control_ge'
        else:
            reason = 'unconfirmed'
        metrics['touch_diag'] = {'reason': reason,
                                 'max_fT': round(max_fT, 3),
                                 'max_sep': round(max_sep, 3),
                                 'n_active': n_active,
                                 'n_watched': len(series),
                                 'frac_grid_missing': round(frac_grid_missing, 3),
                                 'board_occ_peak': round(self.board_occ_peak, 3)}
        # Per-cell occlusion at the board_occ peak (the press): lets store_results find which
        # cell/piece the hand actually touched when the TARGET was not the one touched.
        if self.cell_occ_at_peak is not None:
            metrics['cell_occ_peak'] = [[round(float(v), 3) for v in row] for row in self.cell_occ_at_peak]
            metrics['cell_occ_peak_frame'] = self.cell_occ_at_peak_frame
        if self.cell_touch_score is not None:
            metrics['cell_touch_score'] = [[round(float(v), 3) for v in row] for row in self.cell_touch_score]
        # Re-derive hand_exit from the recorded occlusion curve (touch -> withdrawal): ONE
        # uniform definition for every trial, not just the misses. It removes the
        # source-dependent confirmation lag of the live detection (measured shift vs live:
        # 0..3 frames earlier, i.e. the true crossing). The live frame is kept for audit.
        if confirmed:
            self._posthocBump(metrics)

    def _bumpLandmarks(self, series, enter, anchor=None):
        """Extract the reach bump on an occlusion (frame, value) series: rise (start of the
        climb), peak (max), valley (end of the fall = first return to posthoc_exit_ratio of
        the peak AFTER it, robust to the next trial's re-rise). Returns None if there is no
        real bump (peak below `enter`). `anchor` (the touch frame) bounds the peak search so
        the next trial's rise is not taken as the peak."""
        pts = [(f, v) for f, v in series if v is not None]
        if len(pts) < 6:
            return None
        fr = [f for f, _ in pts]
        vv = [v for _, v in pts]
        hi = (next((i for i, f in enumerate(fr) if f >= anchor), len(vv) - 1)
              if anchor is not None else len(vv) - 1)
        win = vv[:hi + 8]
        peak = max(win, default=0.0)
        if peak < enter:
            return None
        pk = win.index(peak)
        lvl = self.posthoc_exit_ratio * peak
        rs = pk
        while rs > 0 and vv[rs - 1] > lvl:        # walk back the rising edge
            rs -= 1
        valley = next((fr[i] for i in range(pk + 1, len(vv)) if vv[i] <= lvl), None)
        return {'rise': fr[rs], 'peak': fr[pk], 'peak_val': round(peak, 3), 'valley': valley}

    def _posthocBump(self, metrics):
        """Extract the reach landmarks on BOTH curves (target fT and whole-board board_occ)
        and re-derive hand_exit consistently. hand_exit = hand off the BOARD = the board
        bump's valley when there was a real whole-board occlusion (the arm covered the
        board); for a local finger touch (board barely moves) it falls back to the target
        valley, where the finger IS effectively the hand. Records all 6 landmarks in
        metrics['bump'] for audit; keeps the live hand_exit frame."""
        touch = metrics.get('target_touch_capture')
        st = metrics.get('signal_trace', [])
        tgt = self._bumpLandmarks([(s['f'], s.get('fT')) for s in st], self.ft_enter_level, anchor=touch)
        brd = self._bumpLandmarks([(s['f'], s.get('bocc')) for s in st], self.board_occ_enter_level, anchor=touch)
        metrics['bump'] = {'target': tgt, 'board': brd}
        he = (brd['valley'] if (brd is not None and brd['valley'] is not None)
              else (tgt['valley'] if tgt is not None else None))
        if he is not None:
            live, live_src = metrics.get('hand_exit_capture'), metrics.get('hand_exit_source')
            if live is not None and live_src not in (None, 'curve'):
                metrics['hand_exit_live'] = live          # keep the live frame for audit
                metrics['hand_exit_live_source'] = live_src
            metrics['hand_exit_capture'] = he
            metrics['hand_exit_source'] = 'curve_board' if (brd is not None and brd['valley'] is not None) else 'curve_target'
        # touch = the target PEAK (full contact), refined from the live rising-edge frame.
        if tgt is not None:
            if metrics.get('target_touch_capture') is not None:
                metrics['target_touch_live'] = metrics['target_touch_capture']
            metrics['target_touch_capture'] = tgt['peak']
        # motor_onset = hand enters the board. The live contour-loss (board border ArUcos
        # covered) is the natural entry, but it ALSO fires on homography flicker or a hand
        # resting at the extreme edge WITHOUT occluding the cells: measured cases lose the
        # contour while BOTH target (fT) and whole-board (board_occ) occlusion stay ~0 for
        # ~1.7s, then a fast reach spikes them at the touch. So validate the contour-loss
        # with the occlusion: if the earliest real occlusion rise (target or board) follows
        # within motor_onset_artifact_gap it was a real entry and is kept; if it comes much
        # later, the contour-loss had no hand over the board (artifact) and motor_onset is
        # moved to the occlusion rise. The contour frame is kept as motor_onset_live.
        contour = metrics.get('motor_onset_capture')
        rise = min([r for r in ((tgt['rise'] if tgt else None), (brd['rise'] if brd else None))
                    if r is not None], default=None)
        if contour is not None and rise is not None and (rise - contour) > self.motor_onset_artifact_gap:
            metrics['motor_onset_live'] = contour
            metrics['motor_onset_capture'] = rise
            # Classify WHY the contour-loss was spurious, from the trace between the lost
            # contour and the real occlusion rise: a contour dropped while the homography
            # was DOWN or the pose was unreliable (grid from reference) is a homography
            # artifact; otherwise the hand was at the extreme edge without occluding cells.
            win = [s for s in st if contour <= s['f'] < rise]
            bad_h = sum(1 for s in win if not s.get('homog', 1) or s.get('grid') == 'ref')
            metrics['motor_onset_source'] = ('curve_rise_homography'
                                             if win and bad_h >= 0.5 * len(win) else 'curve_rise_edge')
        # Temporal congruence: a real reach has BOTH rises before the touch (fT peak) and
        # BOTH valleys after it. The order WITHIN each pair (board vs target) is free -- it
        # only reflects reach geometry: a close/edge target is reached finger-first, before
        # the arm occludes the board ('finger_led'), recorded as info, NOT an error. (A
        # strict total order wrongly flagged ~24% of normal finger-led reaches.)
        touch_f = tgt['peak'] if tgt else None
        rises = [r for r in ((brd['rise'] if brd else None), (tgt['rise'] if tgt else None)) if r is not None]
        valleys = [v for v in ((tgt['valley'] if tgt else None), he) if v is not None]
        metrics['bump']['congruent'] = (touch_f is None
                                        or (all(r <= touch_f + 3 for r in rises)
                                            and all(touch_f <= v + 3 for v in valleys)))
        if brd and tgt and brd.get('rise') is not None and tgt.get('rise') is not None:
            metrics['bump']['reach_style'] = 'finger_led' if tgt['rise'] < brd['rise'] else 'arm_led'

    """
        Drops gaze samples recorded after end_frame and rebuilds the per color/shape
        counters, so counters, sequence and duration refer to the same time span.
        Only execution-phase samples count: pre_start ones live in the sequence but
        are not part of the per color/shape summary.
    """
    def trimTrialToFrame(self, metrics, end_frame):
        if 'sequence' not in metrics:
            return
        # Trim search/pre_start gaze to the trial end; KEEP the 'motor' samples (reach +
        # withdrawal, recorded after end_frame in motor_recovery) so the trajectory CSV
        # carries the motor/withdraw phases. Only 'execution' samples feed the counters.
        metrics['sequence'] = [s for s in metrics['sequence']
                               if s['frame'] <= end_frame or s.get('phase') == 'motor']
        for key in [k for k in metrics.keys() if k not in self.METADATA_KEYS]:
            del metrics[key]
        # Per-colour search counters: gaze from the start UP TO THE TOUCH (the user-level
        # trial end). That is the 'execution' gaze (search/verification, up to the border
        # crossing) PLUS the 'motor' reach gaze up to the touch frame. Gaze after the touch
        # (withdrawal) and the pre_start/on_panel/blank gaze do NOT count.
        touch = metrics.get('target_touch_capture')
        for s in metrics['sequence']:
            phase = s.get('phase', 'execution')
            counts = (phase == 'execution') or (phase == 'motor' and touch is not None and s['frame'] <= touch)
            if not counts:
                continue
            if s['color'] not in metrics:
                metrics[s['color']] = {s['shape']: {True: 0, False: 0}}
            if s['shape'] not in metrics[s['color']]:
                metrics[s['color']][s['shape']] = {True: 0, False: 0}
            metrics[s['color']][s['shape']][s['slot']] += 1

    """
        Stores one gaze sample in the current trial. Execution samples update the
        per color/shape counters; pre_start ones (gaze on board during the panel
        removal window) only join the sequence, flagged with their phase.
    """
    def _projectGazeCov(self, desnorm_coord, cov_img):
        """Map a per-sample gaze covariance from normalised image coords (top-left) to
        BOARD-NORM coords, via the numerical Jacobian of [correctCoordinates ->
        getPixelBoardNorm] at this gaze point. Returns a 2x2 list or None."""
        if cov_img is None or desnorm_coord is None or self.board_handler.cell_matrix is None:
            return None
        H = self.board_handler.homography
        if H is None:
            return None
        base = np.asarray(desnorm_coord, float).reshape(-1)[:2]

        def fwd(px):
            cc = self.distortion_handler.correctCoordinates(np.array([[px[0], px[1]]]), H)
            return np.asarray(self.board_handler.getPixelBoardNorm(cc.tolist()), float).reshape(-1)[:2]

        try:
            b0 = fwd(base)
            if not np.all(np.isfinite(b0)):
                return None
            eps = 1.0
            jx = (fwd(base + np.array([eps, 0.0])) - b0) / eps
            jy = (fwd(base + np.array([0.0, eps])) - b0) / eps
            J = np.column_stack([jx, jy])
            # norm-image -> desnorm pixel, using the ACTUAL world-frame size (the same one the
            # mean is denormalised with in step()), not a hardcoded 1280x720: otherwise the
            # covariance and its mean live in different pixel spaces on any non-720p video.
            fw, fh = getattr(self, 'world_frame_wh', (1280.0, 720.0))
            Dwh = np.diag([float(fw), float(fh)])
            cov_px = Dwh @ np.asarray(cov_img, float) @ Dwh
            nb = J @ cov_px @ J.T
            if not np.all(np.isfinite(nb)):
                return None
            return [[float(nb[0, 0]), float(nb[0, 1])], [float(nb[1, 0]), float(nb[1, 1])]]
        except Exception:
            return None

    def recordGazeSample(self, capture_idx, color, shape, slot, board_coord, corrected_coord, phase,
                         cov_img=None, desnorm_coord=None):
        if 'sequence' not in self.board_metrics_now:
            self.board_metrics_now['sequence'] = []
        if phase == 'pre_start' and 'early_init_capture' not in self.board_metrics_now:
            self.board_metrics_now['early_init_capture'] = capture_idx

        if phase == 'execution':
            if color not in self.board_metrics_now:
                self.board_metrics_now[color] = {shape: {True: 0, False: 0}}
            if shape not in self.board_metrics_now[color]:
                self.board_metrics_now[color][shape] = {True: 0, False: 0}
            self.board_metrics_now[color][shape][slot] += 1

        # board projection may be unknown for off-board/panel gaze recorded before the
        # board pose is solved: store a null normalized coord rather than dropping the sample
        norm_board_coord = (self.board_handler.getPixelBoardNorm(corrected_coord.tolist()).tolist()
                            if corrected_coord is not None else [None, None])
        # gaze on the detected panel (on_panel) or on a cell covered by a flat-white
        # surface (blank, the panel cardboard sweeping over it) is NOT a visible-cell
        # observation, so it gets no board-cell projection (reads 0% board).
        norm_board_cov = (self._projectGazeCov(desnorm_coord, cov_img)
                          if (corrected_coord is not None and phase not in ('on_panel', 'blank')) else None)
        self.board_metrics_now['sequence'].append({'color': color,
                                                   'shape': shape,
                                                   'slot': slot,
                                                   'frame': capture_idx,
                                                   'phase': phase,
                                                   'board_coord': list(board_coord),
                                                   'norm_board_coord': norm_board_coord,
                                                   'norm_board_cov': norm_board_cov})

    """
        Gaze during the panel removal window: the board pose may be known (arucos +
        session reference grid) before the full border is visible. EVERY sample is
        recorded so analysts get the complete pre-trial view and filter by phase:
          on_panel  - over the sample-panel polygon (looking at the cue)
          not_board - off the board entirely
          blank     - on a board cell still covered by the panel (flat white)
          pre_start - on an exposed board cell (counts once the start is backdated)
        Only pre_start (exposed cell) can later be relabeled to execution and feed the
        per-colour counters; the other three never increment them.
    """
    def processEarlyGaze(self, capture_idx, desnormalized_coord_list):
        if not desnormalized_coord_list:
            return
        panel_polygon = self.panel_handler.getPanelPolygon()

        for index, coordinates in enumerate(desnormalized_coord_list):
            cov_img = self.gaze_cov_list[index] if index < len(self.gaze_cov_list) else None
            und_coord = self.distortion_handler.correctCoordinates(coordinates, homography=None)[0]
            coord_info = self.board_handler.getPixelInfo([coordinates])
            info = coord_info[0] if coord_info else ('not_board', 'not_board', False, [-1, -1], None)
            color, shape, slot, board_coord, corrected_coord = info

            if panel_polygon is not None and \
               cv.pointPolygonTest(panel_polygon, (float(und_coord[0]), float(und_coord[1])), False) >= 0:
                self.recordGazeSample(capture_idx, 'on_panel', 'on_panel', False, [-1, -1], corrected_coord, phase='on_panel',
                                      cov_img=cov_img, desnorm_coord=coordinates)
                self.gaze_classification.append((und_coord, 'on_panel'))
                continue

            if color == 'not_board':
                self.recordGazeSample(capture_idx, 'not_board', 'not_board', False, [-1, -1], corrected_coord, phase='not_board',
                                      cov_img=cov_img, desnorm_coord=coordinates)
                self.gaze_classification.append((und_coord, 'not_board'))
                continue

            if self.board_handler.isRegionBlank(corrected_coord):
                # Flat white where a piece or slot outline should be visible: the panel
                # (or another blank surface) still covers this cell. Keep the cell id.
                self.recordGazeSample(capture_idx, color, shape, slot, board_coord, corrected_coord, phase='blank',
                                      cov_img=cov_img, desnorm_coord=coordinates)
                self.gaze_classification.append((und_coord, 'blank'))
                continue

            self.recordGazeSample(capture_idx, color, shape, slot, board_coord, corrected_coord, phase='pre_start',
                                  cov_img=cov_img, desnorm_coord=coordinates)
            self.gaze_classification.append((und_coord, 'pre_start'))

    """
        The trial end (frame + status) is decided here, but before storing it the
        machine watches the hand leave the board (test_motor_recovery) to record
        hand_exit. The actual storage happens in test_finish_execution_state.
    """
    def finishTrial(self, end_capture, status, capture_idx, reason):
        self.pending_finish = (end_capture, status)
        self.motor_recovery_deadline = capture_idx + self.motor_recovery_max_frames
        self.motor_recovery_streak = 0
        self.motor_recovery_misses = 0
        self.motor_recovery_exit_frame = None
        self.current_state = "test_motor_recovery"
        logStateChange(f"[StateMachine::test_execution] [{capture_idx}] Trial end ({reason}) set to frame {end_capture}. Watching hand exit before closing.")

    """
        After the end is decided, wait until the board contour is back in a sustained
        way (the hand left the board) to record hand_exit, then close. CRITICAL: this
        state must never stop watching for the next panel, or a panel shown during the
        wait would be missed and the sequence would lose sync (it lost ~3% of valid
        trials). A confirmed panel here means the NEXT presentation already started:
        the sample panel is removed at trial start, so by the motor phase it is gone,
        and any confirmed panel (even a re-presentation of the SAME colour/shape, which
        some recordings do and the trial-config buffer slot absorbs) is a new trial.
        On a confirmed panel it yields immediately so init matches it to the right slot;
        otherwise it gives up after a short timeout. Without this, the hand_exit wait
        swallowed a brief same-panel re-presentation and that trial was lost.
    """
    def test_motor_recovery_state(self, undistorted_image, capture_idx, desnormalized_coord_list):
        # processPanel applies the same 4-frame confirmation init uses, so the counter
        # carries over and init picks the panel up without re-counting from scratch.
        next_panel = self.processPanel(undistorted_image, capture_idx, desnormalized_coord_list)
        if next_panel is not None:
            self.current_state = "test_finish_execution"
            return

        self.board_handler.step(undistorted_image, self.corners, self.ids,
                                panel_polygon=self.panel_handler.getPanelPolygon())

        # Keep recording the gaze trajectory through the motor phase (reach + withdrawal),
        # tagged 'motor' so it joins the sequence CSV (phases motor/withdraw) but does NOT
        # enter the per-colour search counters. The board is occluded by the hand here, so
        # the cell is the board-layout cell under the gaze, not necessarily visible.
        for index, coord_data in enumerate(self.board_handler.getPixelInfo(desnormalized_coord_list)):
            color, shape, slot, board_coord, corrected_coord = coord_data
            cov_img = self.gaze_cov_list[index] if index < len(self.gaze_cov_list) else None
            self.recordGazeSample(capture_idx, color, shape, slot, board_coord, corrected_coord, phase='motor',
                                  cov_img=cov_img, desnorm_coord=desnormalized_coord_list[index])

        # Watch the target touch through the motor phase (the hand is over the board now).
        had_touch = 'target_touch_capture' in self.board_metrics_now
        self._trackTargetTouch(capture_idx)
        if not had_touch and 'target_touch_capture' in self.board_metrics_now:
            # Touch just confirmed: RESTART the hand_exit watch so hand_exit becomes the
            # contour return AFTER the touch (the hand actually leaving), not the mid-reach
            # contour return that happens while the hand is still inside reaching (that
            # early return used to close the trial before the touch -> measured 7/7 misses).
            self.motor_recovery_streak = 0
            self.motor_recovery_misses = 0
        touched = 'target_touch_capture' in self.board_metrics_now

        ## hand_exit source 1 (v1.2.0): the LOCAL target occlusion (fT) returning to
        # baseline. More sensitive than the whole-board occlusion for small reaches (a
        # finger over one cell). Requires fT to have clearly risen first (a real reach).
        if self.last_occlusion_measure is not None and 'hand_exit_capture' not in self.board_metrics_now:
            fT = self.last_occlusion_measure[0]
            self.ft_peak = max(self.ft_peak, fT)
            if self.ft_peak >= self.ft_enter_level:
                if fT <= self.ft_exit_level:
                    if self.ft_exit_streak == 0:
                        self.ft_exit_start_frame = capture_idx
                    self.ft_exit_streak += 1
                else:
                    self.ft_exit_streak = 0
                if self.ft_exit_streak >= self.ft_exit_confirm:
                    self.board_metrics_now['hand_exit_capture'] = self.ft_exit_start_frame
                    self.board_metrics_now['hand_exit_source'] = 'ft_return'
                    self.current_state = "test_finish_execution"
                    return

        ## hand_exit source 2: the whole-board occlusion returning to baseline. Unlike
        # the contour path below, it stays high mid-reach (the hand is in the centre), so
        # its sustained return is the hand actually leaving even when no touch was
        # detected. Requires the board to have been clearly occluded first (peak). Falls
        # back to the contour path when the board pose is unavailable (board_occ is None).
        board_occ = self._trackBoardOcclusion(capture_idx)
        self._traceTouch(capture_idx, 'motor_recov')
        if board_occ is not None and self.board_occ_peak >= self.board_occ_enter_level:
            if board_occ <= self.board_occ_exit_level:
                if self.board_occ_exit_streak == 0:
                    self.board_occ_exit_start_frame = capture_idx
                self.board_occ_exit_streak += 1
            else:
                self.board_occ_exit_streak = 0
            if self.board_occ_exit_streak >= self.board_occ_exit_confirm \
               and 'hand_exit_capture' not in self.board_metrics_now:
                self.board_metrics_now['hand_exit_capture'] = self.board_occ_exit_start_frame
                self.board_metrics_now['hand_exit_source'] = 'board_occ'
                self.current_state = "test_finish_execution"
                return

        if self.board_handler.contour_detected_raw:
            if self.motor_recovery_streak == 0:
                self.motor_recovery_exit_frame = capture_idx
            self.motor_recovery_streak += 1
            self.motor_recovery_misses = 0
            # Record hand_exit + close only once the touch is resolved and the hand is
            # back out (contour sustained AFTER the touch). Before the touch, a contour
            # return is the mid-reach one and is ignored, so the touch watch survives.
            if self.motor_recovery_streak >= self.motor_recovery_confirm and touched \
               and 'hand_exit_capture' not in self.board_metrics_now:
                self.board_metrics_now['hand_exit_capture'] = self.motor_recovery_exit_frame
                self.board_metrics_now['hand_exit_source'] = 'contour'
                self.current_state = "test_finish_execution"
        elif self.motor_recovery_streak > 0:
            # Tolerate brief contour flicker as the hand withdraws: only reset the
            # streak after a few consecutive misses (a single dropped frame used to
            # reset it and lose hand_exit).
            self.motor_recovery_misses += 1
            if self.motor_recovery_misses > self.motor_recovery_miss_tolerance:
                self.motor_recovery_streak = 0
                self.motor_recovery_misses = 0

        # Deadline fallback (no touch detected, or the hand never left cleanly): close,
        # recording the best hand_exit estimate available. The next-panel case is at the top.
        if capture_idx >= self.motor_recovery_deadline:
            if 'hand_exit_capture' not in self.board_metrics_now and self.motor_recovery_streak > 0:
                self.board_metrics_now['hand_exit_capture'] = self.motor_recovery_exit_frame
                self.board_metrics_now['hand_exit_source'] = 'deadline'
            self.current_state = "test_finish_execution"

    def test_finish_execution_state(self, undistorted_image, capture_idx, desnormalized_coord_list):

        # End frame and status decided by whichever criterion fired (finishTrial);
        # default to the backdated board loss for safety
        if self.pending_finish is not None:
            end_capture, status = self.pending_finish
        else:
            end_capture = self.last_raw_contour_frame if self.last_raw_contour_frame is not None else capture_idx
            status = 'test_finish_execution'
        self.board_metrics_now['end_capture'] = end_capture
        self.trimTrialToFrame(self.board_metrics_now, end_capture)
        self.board_metrics_now['status'] = status
        self._deriveTouchDiagnostics(self.board_metrics_now)
        self.board_metrics_store[(self.block_id, self.trial_id)] = {f"{self.current_test_key['color']}_{self.current_test_key['shape']}": copy.deepcopy(self.board_metrics_now)}
        self.board_metrics_store['latest'] = self.board_metrics_store[(self.block_id, self.trial_id)]
        self.resetTrialTracking()

        self.current_state = "init"
        logStateChange(f"[StateMachine::test_finish_execution::] Switch to init. Trial closed at frame {end_capture} with status {status}.")

    ## Clears all the per-trial tracking state
    def resetTrialTracking(self):
        self.board_metrics_now = {}
        self.current_test_key = None
        self.last_raw_contour_frame = None
        self.pending_finish = None
        self.last_occlusion_measure = None
        self.target_occlusion_counter = 0
        self.target_occlusion_start_frame = None
        self.target_warmup_end = None
        self.target_tracking_active = False
        self.target_is_warm = False
        self.touch_threshold = self.target_occlusion_threshold
        self.motor_recovery_deadline = None
        self.motor_recovery_streak = 0
        self.motor_recovery_misses = 0
        self.motor_recovery_exit_frame = None
        self.board_occ_active = False
        self.board_occ_peak = 0.0
        self.cell_occ_at_peak = None   # per-cell occlusion snapshot at the board_occ peak (touched cell)
        self.cell_occ_at_peak_frame = None
        self.cell_touch_score = None   # accumulated per-cell FOCAL occlusion (sustained fingertip)
        self.board_occ_exit_streak = 0
        self.board_occ_exit_start_frame = None
        self.last_board_occ = None
        self.ft_peak = 0.0
        self.ft_exit_streak = 0
        self.ft_exit_start_frame = None
        self.occlusion_series = []
        self.board_contour_nondetected_counter = 0
        self.board_contour_detected_counter = 0
        self.contour_streak_start_frame = None
        self.board_handler.clearTargetTracking()

    # If end of video was detected close latest test
    def handle_end_of_video(self):

        if self.current_state != "init":
            # Backdate to the last frame the board was actually seen, instead of
            # dragging the trial until the very last frame of the recording
            end_capture = self.last_raw_contour_frame if self.last_raw_contour_frame is not None else self.last_frame_number
            self.board_metrics_now['end_capture'] = end_capture
            self.trimTrialToFrame(self.board_metrics_now, end_capture)
            if 'init_capture' in self.board_metrics_now:
                # Trial was running with gathered data: close it as valid with a
                # distinctive status (the recording may have stopped before the
                # motor response was completed)
                self.board_metrics_now['status'] = 'test_finish_by_end_of_video'
                self._deriveTouchDiagnostics(self.board_metrics_now)
                key = f"{self.current_test_key['color']}_{self.current_test_key['shape']}"
            else:
                key = f"end_of_video_error_{self.current_test_key['color']}_{self.current_test_key['shape']}"
            self.board_metrics_store[(self.block_id, self.trial_id)] = {key: copy.deepcopy(self.board_metrics_now)}
            self.board_metrics_store['latest'] = self.board_metrics_store[(self.block_id, self.trial_id)]
            self.resetTrialTracking()

            self.current_state = "init"

    def load_from_yaml(self, output_path, participant_id = ""):
        data_store = parseYaml(os.path.join(output_path,f'data_{participant_id}.yaml'))

        self.video_fps = data_store['video_fps']
        participant_id = data_store['participant_id']
        self.frame_data_store = data_store['frames_info']
        self.fixation_data_store = data_store['fixations_info']
        self.board_metrics_store = data_store['trials_data']

    def load_from_pickle(self, output_path, participant_id = ""):
        with open(os.path.join(output_path,f'data_{participant_id}.pkl'), 'rb') as f:
            data_store = pickle.load(f)

        self.video_fps = data_store['video_fps']
        participant_id = data_store['participant_id']
        self.frame_data_store = data_store['frames_info']
        self.fixation_data_store = data_store['fixations_info']
        self.board_metrics_store = data_store['trials_data']
        # state_transitions is written by store_results: restore it so a reload+store
        # round-trip (e.g. the post-hoc landmark re-processing tool) keeps the timeline.
        self.state_transitions = data_store.get('state_transitions', self.state_transitions)
        # Preserve the original execution-mode provenance across a reload+store round-trip.
        self.slow_analysis = data_store.get('slow_analysis', self.slow_analysis)

    def _idtFixations(self, pts):
        """I-DT fixations over gaze samples (each with norm_board_coord, board_coord, frame):
        maximal runs whose normalized bounding box stays under the dispersion bound and last at
        least the min-samples. Returns [(start_frame, end_frame, (col,row) predominant cell)].
        Same detector as target_found; reused for the gaze 'validation' (last fixation before
        the touch = the piece the eyes committed to)."""
        disp_max = self.target_found_fixation_dispersion
        min_n = self.target_found_min_fixation_samples
        out, i = [], 0
        while i < len(pts):
            xs = [pts[i]['norm_board_coord'][0]]
            ys = [pts[i]['norm_board_coord'][1]]
            j = i + 1
            while j < len(pts):
                x, y = pts[j]['norm_board_coord']
                if (max(xs + [x]) - min(xs + [x])) + (max(ys + [y]) - min(ys + [y])) > disp_max:
                    break
                xs.append(x); ys.append(y); j += 1
            if (j - i) >= min_n:
                cells = Counter(tuple(pts[k]['board_coord']) for k in range(i, j)
                                if pts[k].get('board_coord') and pts[k]['board_coord'][0] is not None)
                cell = cells.most_common(1)[0][0] if cells else None
                out.append((pts[i]['frame'], pts[j - 1]['frame'], cell))
                i = j
            else:
                i += 1
        return out

    def _deriveWrongPiece(self):
        """Per-trial: the piece the hand TOUCHED, the piece the GAZE validated, and the error
        type -- stored in the trial metrics BEFORE serialization so the PKL (the source of
        truth) is complete; the CSVs are just a projection of it.
          touched  = the target if its touch was confirmed; else the cell with the most
                     occlusion at the board_occ peak (the press), if above wrong_touch_min_occ.
          validated= the cell of the last BOARD fixation (I-DT) before the touch / trial end.
          error    = correct (touched the target) | perceptual (gaze and touch on the SAME
                     wrong piece) | motor (touched a piece other than the validated/target one)."""
        # Board layout (cell -> piece), fixed for the session, from all on-board gaze samples.
        cell_piece = {}
        for tm in self.board_metrics_store.values():
            if not isinstance(tm, dict):
                continue
            for m in tm.values():
                if not isinstance(m, dict):
                    continue
                for s in m.get('sequence', []):
                    bc, col = s.get('board_coord'), s.get('color')
                    if bc and bc[0] is not None and col and col not in ('not_board', 'on_panel'):
                        cell_piece[(int(bc[0]), int(bc[1]))] = f"{col}_{s.get('shape')}"
        for key, tm in self.board_metrics_store.items():
            if key == 'latest':
                continue
            for name, m in tm.items():
                if not isinstance(m, dict) or m.get('init_capture', -1) == -1 \
                   or name.startswith(('missing', 'transition', 'end_of')):
                    continue
                target_cord = m.get('target_cord')
                target_cell = (int(target_cord[1]), int(target_cord[0])) if (target_cord and target_cord[0] is not None) else None
                touch = m.get('target_touch_capture')
                reach = m.get('motor_onset_capture')
                touched_cell = touched_piece = wrong_touch_frame = None
                if touch is not None and target_cell is not None:
                    touched_cell, touched_piece = target_cell, name
                elif m.get('cell_touch_score') is not None and reach is not None:
                    # Touched cell = the SUSTAINED-FOCAL maximum: the cell where a fingertip
                    # rested (focal every frame) rather than where the arm merely swept. Gaze is
                    # NOT used here, on purpose (its agreement/disagreement is the error signal).
                    score = np.array(m['cell_touch_score'], dtype=float)
                    r, c = np.unravel_index(int(score.argmax()), score.shape)
                    if score[r, c] >= self.wrong_touch_min_score:
                        touched_cell, touched_piece = (int(c), int(r)), cell_piece.get((int(c), int(r)))
                        wrong_touch_frame = m.get('cell_occ_peak_frame')
                # gaze validation: last BOARD fixation (excludes panel / off-board) before the touch
                board_pts = [s for s in m.get('sequence', [])
                             if s.get('norm_board_coord') and s['norm_board_coord'][0] is not None
                             and s.get('color') not in ('not_board', 'on_panel')]
                boundary = touch if touch is not None else m.get('end_capture')
                gaze_cell = gaze_piece = frame_validation = None
                before = [f for f in self._idtFixations(board_pts) if f[0] <= boundary and f[2] is not None]
                if before:
                    frame_validation = before[-1][0]
                    gaze_cell = (int(before[-1][2][0]), int(before[-1][2][1]))
                    gaze_piece = cell_piece.get(gaze_cell)
                # error_type from RELIABLE signals only. The touched_cell/piece are kept as
                # EXPERIMENTAL info (too noisy to trust -- the occlusion cannot yet separate the
                # fingertip from the arm; see docs) and do NOT drive this. A trial is flagged when
                # the hand both ENTERED and LEFT the board (a completed reach) but neither the
                # target was touched NOR the gaze committed to it: something off-target happened
                # (which piece it was is not yet located -- review it in the debug figure).
                target_touched = touch is not None
                reach_done = reach is not None and m.get('hand_exit_capture') is not None
                gaze_on_target = target_cell is not None and gaze_cell == target_cell
                if target_touched:
                    error_type = 'correct'
                elif reach_done and target_cell is not None and not gaze_on_target:
                    error_type = 'off_target'       # reached+withdrew, target neither touched nor looked at
                elif reach_done and gaze_on_target:
                    error_type = 'no_touch'          # looked at the target, reached, but no touch confirmed
                else:
                    error_type = None
                m['touched_cell'], m['touched_piece'] = touched_cell, touched_piece
                m['gaze_validated_cell'], m['gaze_validated_piece'] = gaze_cell, gaze_piece
                m['frame_validation'], m['wrong_touch_frame'] = frame_validation, wrong_touch_frame
                m['error_type'] = error_type

    def store_results(self, output_path, participant_id = "", video_fps = None, run_config = None):

        self.handle_end_of_video()
        self.board_metrics_store.pop('latest', None)
        # Derive the touched/validated piece + error type INTO the metrics before serialising,
        # so the PKL is complete (the CSVs below are only a projection of it).
        self._deriveWrongPiece()

        # Provenance: every output records HOW it was produced, so a result is never silently
        # mode-dependent and nobody has to reverse-engineer it later. slow_analysis is the
        # authoritative one; run_config carries the rest of the quality-affecting options.
        run_config = dict(run_config or {})
        run_config.setdefault('slow_analysis', self.slow_analysis)
        partial = bool(run_config.get('start_frame') or run_config.get('end_frame') is not None)

        # Two guards that make a degraded/partial output SCREAM here instead of being
        # discovered weeks later by comparing versions:
        n_missing = sum(1 for k, t in self.board_metrics_store.items()
                        if k != 'latest' and list(t.keys())[0].startswith('missing_trial_error'))
        n_total = sum(1 for k in self.board_metrics_store if k != 'latest')
        if partial:
            log(f"[store_results::{participant_id}] NOTE: PARTIAL run (start_frame={run_config.get('start_frame')}, "
                f"end_frame={run_config.get('end_frame')}) -- this output is a debug SEGMENT, not the full "
                f"recording; do not use it as a complete result.")
        elif n_total and n_missing > max(2, 0.05 * n_total):
            log(f"[store_results::{participant_id}] WARNING: {n_missing}/{n_total} expected trials were NOT "
                f"detected (slow_analysis={self.slow_analysis}). A high miss rate often means the fast path "
                f"subsampled past marginal panels -- re-run with slow_analysis before trusting this output.")

        gaze_sampling_rate = getattr(self.eye_data_handler, 'gaze_sampling_rate', None)
        data_store = {'sw_version': __version__, 'slow_analysis': self.slow_analysis, 'run_config': run_config, 'video_fps': video_fps, 'gaze_sampling_rate': gaze_sampling_rate, 'participant_id': participant_id, 'frames_info': self.frame_data_store, 'fixations_info': self.fixation_data_store, 'trials_data': self.board_metrics_store, 'state_transitions': self.state_transitions}
        with open(os.path.join(output_path,f'data_{participant_id}.pkl'), 'wb') as f:
            pickle.dump(data_store, f)
        
        dumpYaml(os.path.join(output_path,f'data_{participant_id}.yaml'), data_store, 'w')

        ## CSV trials
        csv_data = []
        csv_data_seq = []
        # Two blocks of columns: derived per-phase durations (convenient) and the raw
        # event frames in the World video (so any analysis can recompute its own
        # intervals / pick its own trial-end point). Empty when the event was not seen.
        # v1.2.0 clean model: the trial is RE-BASED to start at the search onset (panel
        # removal / first board gaze), not the full-board contour confirmation, so
        # trial_duration_s INCREASES vs v1.0/v1.1 (not comparable -- by design). Marks
        # are anchored on the homography+occlusion. New per-phase durations and
        # covariates (anticipation, reach distance) are added for the analysis team.
        csv_data.append(['block_index', 'trial_index', 'trial_name', 'Color', 'Shape', 'Piece Fixations', 'Slot only Fixations',
                         'trial_duration_s', 'time_to_target_s', 'target_found_confidence', 'search_duration_s', 'reach_duration_s', 'withdraw_duration_s',
                         'frame_search_start', 'frame_init', 'frame_target_found', 'frame_validation', 'frame_motor_onset', 'frame_target_touch', 'frame_hand_exit', 'frame_end',
                         'anticipatory_gaze', 'anticipation_lead_s', 'target_row', 'target_col',
                         'touched_piece', 'touched_cell', 'gaze_validated_piece', 'gaze_validated_cell', 'error_type',
                         'Finish Status'])
        csv_data_seq.append(['block_index', 'trial_index', 'trial_name', 'Color', 'Shape', 'Piece=1/Slot=0', 'Phase', 'Frame_N', 'trial_duration_s', 'Board Coord', 'Board norm Coord', 'Cell Dist', 'Onboard Mass', 'Finish Status'])
        # Behavioural marks collected per trial; merged with the state transitions below
        # into ONE chronological timeline (the unified events CSV the analysts asked for).
        mark_events = []
        for block_id, trial_id in sorted(self.board_metrics_store.keys()):
            trial_metric = self.board_metrics_store[(block_id, trial_id)]
            board_metrics = list(trial_metric.values())[0]
            board_test_name = list(trial_metric.keys())[0]
            if not 'init_capture' in board_metrics or not 'end_capture' in board_metrics:
                continue
            init_capture = board_metrics['init_capture']
            end_capture = board_metrics['end_capture']
            # RE-BASED start: the trial begins at the search onset -- the first board
            # gaze during the panel removal (early gaze) -- or the full-board start if
            # there was no anticipatory gaze. trial_duration is measured from here.
            search_start = board_metrics.get('early_init_capture', init_capture)
            duration_s = (end_capture-search_start)/self.video_fps
            motor_onset = board_metrics.get('motor_onset_capture')          # hand crosses the board border (contour)
            reach_onset = motor_onset
            touch = board_metrics.get('target_touch_capture')
            hand_exit = board_metrics.get('hand_exit_capture')
            # Anticipation covariate: gaze already on the board while the sample panel was
            # still being removed (pre_start samples), and how early it started.
            anticipatory_gaze = sum(1 for s in board_metrics.get('sequence', []) if s.get('phase') == 'pre_start')
            anticipation_lead_s = 0
            if 'early_init_capture' in board_metrics:
                anticipation_lead_s = max(0, (init_capture-board_metrics['early_init_capture'])/self.video_fps)
            # First gaze on the target cell (when it was first found with the eyes).
            # target_cord is [row,col]; sequence board_coord is [col,row]
            first_target_frame = None
            target_found_confidence = ''     # graded: mass of the gaze ellipse on the target cell
            target_cord = board_metrics.get('target_cord')
            if target_cord and target_cord[0] is not None:
                target_colrow = [int(target_cord[1]), int(target_cord[0])]
                ncols, nrows = self.board_handler.board_size[0], self.board_handler.board_size[1]
                # I-DT fixation detection: maximal runs of on-board gaze whose
                # bounding box stays within the dispersion bound; the first such
                # fixation with a majority of samples on the target marks the frame.
                pts = [s for s in board_metrics.get('sequence', [])
                       if s.get('norm_board_coord') and s['norm_board_coord'][0] is not None]
                disp_max = self.target_found_fixation_dispersion
                min_n = self.target_found_min_fixation_samples
                i = 0
                tf_conf = 0.0
                while i < len(pts):
                    xs = [pts[i]['norm_board_coord'][0]]
                    ys = [pts[i]['norm_board_coord'][1]]
                    j = i + 1
                    while j < len(pts):
                        x, y = pts[j]['norm_board_coord']
                        if (max(xs + [x]) - min(xs + [x])) + (max(ys + [y]) - min(ys + [y])) > disp_max:
                            break
                        xs.append(x); ys.append(y); j += 1
                    if (j - i) >= min_n:
                        # mean per-sample ellipse mass on the target cell over this fixation.
                        # target_found_confidence is the max across the trial's fixations;
                        # found = the FIRST fixation whose mean mass reaches the threshold
                        # (uncertainty-aware, not a hard discrete cell vote).
                        masses = []
                        for k in range(i, j):
                            cov = pts[k].get('norm_board_cov')
                            if cov is not None:
                                nb = pts[k]['norm_board_coord']
                                masses.append(_cell_mass(nb[0], nb[1], cov,
                                                         target_colrow[0], target_colrow[1], ncols, nrows))
                        if masses:
                            mean_mass = sum(masses) / len(masses)
                            tf_conf = max(tf_conf, mean_mass)
                            if mean_mass >= self.target_found_mass_threshold and first_target_frame is None:
                                first_target_frame = pts[i]['frame']
                        else:
                            # no uncertainty model on this fixation: fall back to the discrete majority vote
                            on = sum(1 for k in range(i, j) if list(pts[k]['board_coord']) == target_colrow)
                            if on * 2 >= (j - i) and first_target_frame is None:
                                first_target_frame = pts[i]['frame']
                        i = j
                    else:
                        i += 1
                if any(s.get('norm_board_cov') for s in pts):
                    target_found_confidence = round(tf_conf, 3)
            time_to_target_s = '' if first_target_frame is None else max(0, (first_target_frame-search_start)/self.video_fps)
            # Per-phase durations (clean model): search (board gaze until the hand is over
            # the board), reach (hand over board until the touch), withdraw (touch until
            # the hand leaves). Empty when the bounding mark was not observed.
            search_duration_s, reach_duration_s, withdraw_duration_s = '', '', ''
            if reach_onset is not None:
                search_duration_s = max(0, (reach_onset-search_start)/self.video_fps)
            if reach_onset is not None and touch is not None:
                reach_duration_s = max(0, (touch-reach_onset)/self.video_fps)
            if touch is not None and hand_exit is not None:
                withdraw_duration_s = max(0, (hand_exit-touch)/self.video_fps)
            # Target position (row 0 = far/top, board_size[1]-1 = near/bottom; col 0..7).
            # The reach DISTANCE (mm) is a fixed property of the cell (same for every
            # participant), so it is NOT repeated here: it lives in the per-board reference
            # CSV target_geometry.csv (join by trial_name / row,col). See docs.
            target_row, target_col = '', ''
            if target_cord and target_cord[0] is not None:
                target_row, target_col = int(target_cord[0]), int(target_cord[1])

            # Touched piece / gaze validation / error type: READ from the metrics (computed in
            # _deriveWrongPiece before serialisation -- the PKL is the source of truth; this CSV
            # is a projection). Cells are (col,row) tuples -> "row,col" strings for the CSV.
            touched_piece = board_metrics.get('touched_piece') or ''
            gaze_validated_piece = board_metrics.get('gaze_validated_piece') or ''
            error_type = board_metrics.get('error_type') or ''
            frame_validation = board_metrics.get('frame_validation')
            wrong_touch_frame = board_metrics.get('wrong_touch_frame')
            _tc, _vc = board_metrics.get('touched_cell'), board_metrics.get('gaze_validated_cell')
            tc = f"{_tc[1]},{_tc[0]}" if _tc else ''
            vc = f"{_vc[1]},{_vc[0]}" if _vc else ''

            # Behavioural marks of this trial as point events, to be interleaved with the
            # state transitions in the unified timeline. Skipped when not observed; the
            # missing_trial_error placeholders (init/end = -1) carry no real frames.
            if end_capture != -1 and init_capture != -1:
                # NOTE: no 'wrong_touch' mark -- the touched cell is experimental/unreliable, so
                # publishing it as a timeline event would mislead. The off_target anomaly shows as
                # validation (where the eyes went) + motor_onset/hand_exit WITHOUT a target_touch.
                for mark_name, mark_frame in [('search_start', search_start),
                                              ('target_found', first_target_frame),
                                              ('validation', frame_validation),
                                              ('motor_onset', motor_onset),
                                              ('target_touch', touch),
                                              ('hand_exit', hand_exit),
                                              ('trial_end', end_capture)]:
                    if mark_frame is None or mark_frame == '':
                        continue
                    mark_events.append({'block': block_id, 'trial': trial_id, 'trial_name': board_test_name,
                                        'frame': mark_frame, 'event': mark_name,
                                        'from_state': '', 'to_state': ''})

            # Cognitive phase of a given gaze frame, for the sequence CSV
            # Cognitive/motor phase of each gaze sample. The phase boundaries ARE the event
            # marks, so the Phase column alone encodes them (no need to repeat the frames):
            #   pre_start    (gaze during panel removal)
            #   search       (looking, target not yet found)  -> verification at target_found
            #   verification (target found, hand not yet in)  -> motor at motor_onset
            #   motor        (hand reaching the piece)         -> withdraw at the touch
            #   withdraw     (after the touch, hand leaving)   -> trial closes at hand_exit
            # The motor/withdraw gaze is recorded through the motor-recovery window.
            def phaseOf(frame, base_phase):
                # Early-window location tags (on_panel/blank/not_board) and pre_start are
                # kept as-is; only the temporal execution samples are split by the marks.
                if base_phase in ('pre_start', 'on_panel', 'blank', 'not_board'):
                    return base_phase
                if touch is not None and frame >= touch:
                    return 'withdraw'
                if reach_onset is not None and frame >= reach_onset:
                    return 'motor'
                if frame_validation is not None and frame >= frame_validation:
                    return 'validation'            # gaze committed to a piece, hand not in yet
                if first_target_frame is not None and frame >= first_target_frame:
                    return 'verification'
                return 'search'

            # Raw event frames (empty string when the event was not observed)
            f_target = first_target_frame if first_target_frame is not None else ''
            f_valid = frame_validation if frame_validation is not None else ''
            f_motor = motor_onset if motor_onset is not None else ''
            f_touch = touch if touch is not None else ''
            f_exit = hand_exit if hand_exit is not None else ''

            for color, color_item in board_metrics.items():
                if color in self.METADATA_KEYS:
                    continue
                for shape, shape_item in color_item.items():
                    csv_data.append([block_id, trial_id, board_test_name, color, shape, shape_item[True], shape_item[False],
                                     duration_s, time_to_target_s, target_found_confidence, search_duration_s, reach_duration_s, withdraw_duration_s,
                                     search_start, init_capture, f_target, f_valid, f_motor, f_touch, f_exit, end_capture,
                                     anticipatory_gaze, anticipation_lead_s, target_row, target_col,
                                     touched_piece, tc, gaze_validated_piece, vc, error_type,
                                     board_metrics['status']])

            ncols_s, nrows_s = self.board_handler.board_size[0], self.board_handler.board_size[1]
            for step in board_metrics['sequence']:
                nb = step.get('norm_board_coord')
                mx = nb[0] if nb and nb[0] is not None else None
                my = nb[1] if nb and nb[1] is not None else None
                if step.get('phase') in ('on_panel', 'blank'):
                    cell_dist, onboard_mass = '', 0.0   # panel / covered cell -> not a visible-cell observation
                else:
                    cell_dist, onboard_mass = _cell_distribution(mx, my, step.get('norm_board_cov'), ncols_s, nrows_s)
                csv_data_seq.append([
                    block_id,
                    trial_id,
                    board_test_name,
                    step['color'],
                    step['shape'],
                    step['slot'],
                    phaseOf(step['frame'], step.get('phase', 'execution')),
                    step['frame'],
                    duration_s,
                    step['board_coord'],
                    step['norm_board_coord'],
                    cell_dist,
                    onboard_mass,
                    board_metrics['status']
                ])

        with open(os.path.join(output_path,f'trials_data_{participant_id}.csv'), mode="w", newline="") as file:
            csv.writer(file).writerows(csv_data)
        
        with open(os.path.join(output_path,f'trials_data_{participant_id}_sequence.csv'), mode="w", newline="") as file:
            csv.writer(file).writerows(csv_data_seq)

        ## CSV unified timeline: state transitions AND behavioural marks (target_found,
        # motor_onset, target_touch, hand_exit, trial_end, search_start) interleaved in
        # ONE chronological table per trial. Each row carries the exact World frame, its
        # time in seconds, the block/trial, and the event:
        #   - state change -> event='state_change', from_state/to_state filled
        #   - mark         -> event=<mark name>, from_state/to_state empty
        # Filter event=='state_change' for the pure state machine, or the mark names for
        # the behavioural marks; read all rows (sorted by frame) for the full timeline.
        fps = video_fps or self.video_fps
        name_by_trial = {key: list(tm.keys())[0] for key, tm in self.board_metrics_store.items()}
        events = list(mark_events)
        for t in self.state_transitions:
            events.append({'block': t['block'], 'trial': t['trial'], 'trial_name': t['trial_name'],
                           'frame': t['frame'], 'event': 'state_change',
                           'from_state': t['from_state'], 'to_state': t['to_state']})
        # Stable chronological order within each trial (None block/trial sort first)
        events.sort(key=lambda e: (e['block'] if e['block'] is not None else -1,
                                   e['trial'] if e['trial'] is not None else -1,
                                   e['frame']))
        csv_transitions = [['block_index', 'trial_index', 'trial_name', 'frame', 'time_s',
                            'event', 'from_state', 'to_state']]
        for e in events:
            block = '' if e['block'] is None else e['block']
            trial = '' if e['trial'] is None else e['trial']
            name = e['trial_name'] or name_by_trial.get((e['block'], e['trial']), '')
            time_s = round(e['frame'] / fps, 3) if fps else ''
            csv_transitions.append([block, trial, name, e['frame'], time_s,
                                    e['event'], e['from_state'], e['to_state']])
        with open(os.path.join(output_path,f'trials_data_{participant_id}_transitions.csv'), mode="w", newline="") as file:
            csv.writer(file).writerows(csv_transitions)

        terminal_log = self.print_results()
        with open(os.path.join(output_path,f'result_log_{participant_id}.txt'), 'w') as file:
            file.write(terminal_log)


    def print_results(self):
        
        ## logs fixation data :)
        total_fixations = self.fixation_data_store['total']
        frames_with_fixation = (total_fixations/self.frame_data_store['total'])*100
        
        terminal_log = str()
        terminal_log += bufferStateChangeMsg(f"#############################")
        terminal_log += bufferStateChangeMsg(f"##      Result report      ##")
        terminal_log += bufferStateChangeMsg(f"#############################\n")

        ## logs table with frame distribution
        total_frames = self.frame_data_store['total']
        terminal_log += bufferStateChangeMsg(f"Frames distribution of the {total_frames} fixations involved.")
        terminal_log += bufferMsg(f"* Please note speed multiplier")
        terminal_log += bufferMsg(f"\t· Analyzed from frame {self.init_frame_number} to {self.last_frame_number}")
        terminal_log += bufferMsg(f"\t· Total frames: {self.frame_data_store['total']}")
        terminal_log += bufferMsg(f"\t· Total time (s): {self.frame_data_store['total']/self.video_fps}")
        log_table_data = []
        log_table_headers = ['State Name', 'N Frames', 'Percent.', 'Time (s)']
        for key, item in self.frame_data_store.items():
            if key in ['total']:
                continue
            log_table_data.append([key,item,f"{item/total_frames*100:3f}", item/self.video_fps])
        
        terminal_log += bufferMsg(tabulate(log_table_data, headers=log_table_headers, tablefmt="pretty"))
        terminal_log += bufferMsg("\n\n")


        ## logs table with fixation distribution
        terminal_log += bufferStateChangeMsg(f"Fixation distribution of the {total_fixations} fixations involved. Frames with fixation data: {frames_with_fixation:3f}%")
        terminal_log += bufferMsg(f"* Please note speed multiplier")
        log_table_data = []
        log_table_headers = ['State Name', 'N Fixations', 'Percent.']
        for key, item in self.fixation_data_store.items():
            if key in ['total']:
                continue
            log_table_data.append([key,item,f"{item/total_fixations*100:3f}"])
        
        terminal_log += bufferMsg(tabulate(log_table_data, headers=log_table_headers, tablefmt="pretty"))
        terminal_log += bufferMsg("\n\n")


        ## logs data of each trial
        for block_id, trial_id in sorted(self.board_metrics_store.keys()):
            trial_metric = self.board_metrics_store[(block_id, trial_id)]
            board_metrics = list(trial_metric.values())[0]

            if not 'init_capture' in board_metrics or not 'end_capture' in board_metrics:
                continue    
            test_tag = f"[Block {block_id}][Trial {trial_id}] Search for {list(trial_metric.keys())[0]}"

            log_table_headers = ['Color', 'Shape', 'Piece Fixations', 'Slot only Fixations', 'Finish Status']
            log_table_data = []
            for color, color_item in board_metrics.items():
                if color in self.METADATA_KEYS:
                    continue
                for shape, shape_item in color_item.items():
                    log_table_data.append([color, shape, shape_item[True], shape_item[False], board_metrics['status']])

            formatted_table = tabulate(log_table_data, headers=log_table_headers, tablefmt="pretty")
            
            table_width = len(formatted_table.splitlines()[1]) # Get length from dashes, which is second one
            title_dashes = '-' * ((table_width - len(test_tag)) // 2)

            duration_s = (board_metrics['end_capture']-board_metrics['init_capture'])/self.video_fps
            terminal_log += bufferStateChangeMsg(f"{title_dashes} {test_tag} {title_dashes}")
            terminal_log += bufferMsg(f"    - Started at {board_metrics['init_capture']} frame.")
            terminal_log += bufferMsg(f"    - Ended at {board_metrics['end_capture']} frame.")
            terminal_log += bufferMsg(f"    - Took {board_metrics['end_capture']-board_metrics['init_capture']} frames. ({duration_s} s)")
            for line in formatted_table.splitlines():
                terminal_log += bufferMsg(line)
            terminal_log += bufferMsg("\n\n")


        log(terminal_log)
        return terminal_log
