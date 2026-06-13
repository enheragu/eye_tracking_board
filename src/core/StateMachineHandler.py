#!/usr/bin/env python3
# encoding: utf-8
import os
import copy

import math
from collections import deque
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

        self.board_handler = board_handler
        self.panel_handler = panel_handler
        self.eye_data_handler = eye_data_handler
        self.distortion_handler = distortion_handler

        # Whitelist of all configured marker ids (board + every panel). Detections
        # with any other id are spurious (the detector sometimes fires id 0 on board
        # pieces); they are dropped before any processing or drawing.
        self.valid_aruco_ids = set(self.board_handler.aruco_board_handler.config_ids)
        for panel in self.panel_handler.panel_handler_list:
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
        self.panel_detected_threshold = 2

        ## Target-touch MARK (best-effort, does NOT close the trial): records when the
        # hand TOUCHES the target piece (it is not removed from the board). The touch is
        # a subtle, partial occlusion, so thresholds are moderate; the double-margin
        # requirement keeps false positives down.
        self.target_occlusion_threshold = 0.20   # fraction of changed px in target area
        self.target_occlusion_separation = 0.10  # margin over the control (global) change
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

        ## Motor recovery: after the trial end is decided, keep watching until the
        # board contour comes back in a sustained way (the hand left the board) to
        # record hand_exit, then close. Confirms the touch was final (hand withdrew)
        # and gives the withdrawal time.
        self.motor_recovery_max_frames = 75   # keep watching touch+hand_exit (~2.5s); the
        # touch can peak up to ~57 frames after the border crossing (measured)
        self.motor_recovery_confirm = 3       # sustained contour = hand out of board
        self.motor_recovery_miss_tolerance = 2  # contour flickers as the hand withdraws;
        # a couple of dropped frames must not reset the streak (that lost hand_exit)
        self.motor_recovery_deadline = None
        self.motor_recovery_streak = 0
        self.motor_recovery_misses = 0
        self.motor_recovery_exit_frame = None
        self.pending_finish = None
        self.last_occlusion_measure = None

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

        ## Gaze samples over the camera view, color-coded by how they were used:
        # green=counted (execution), orange=pre_start, red=discarded over the panel,
        # gray=discarded as blank (panel covering), white=not_board, blue=not processed
        GAZE_COLORS = {'execution': (0,255,0), 'pre_start': (0,165,255), 'on_panel': (0,0,255),
                       'blank': (160,160,160), 'not_board': (255,255,255), 'unprocessed': (255,200,0)}
        if self.norm_coord_list and self.board_handler.display_fixation:
            classification = self.gaze_classification
            if not classification:
                # States that do not project gaze (init/get_test_name): show it anyway
                classification = [(self.distortion_handler.correctCoordinates(coord, homography=None)[0], 'unprocessed')
                                  for coord in self.desnormalized_coord_list]
            for und_coord, kind in classification:
                if und_coord[0] < 0 or und_coord[1] < 0:
                    continue
                cv.circle(canvas, (int(und_coord[0]), int(und_coord[1])), radius=5,
                          color=GAZE_COLORS.get(kind, (255,255,255)), thickness=-1)

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
        current_panel = None
        self.panel_handler.step(undistorted_image, self.corners, self.ids)

        shape, aruco, panel = self.panel_handler.getPixelInfo(desnormalized_coord_list)
        current_detected_panel = self.panel_handler.getCurrentPanel()

        if current_detected_panel is None:
            self.panel_detected_counter = []
            return current_panel
        
        if self.panel_detected_counter == []:
            self.panel_detected_counter.append(current_detected_panel)
        elif IsSamePanel(current_detected_panel, self.panel_detected_counter[-1]):
            self.panel_detected_counter.append(current_detected_panel)
        
        if len(self.panel_detected_counter) >= self.panel_detected_threshold:
            current_panel = current_detected_panel

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

        ## Undistort and detect arucos once per frame; both image and detections are
        # shared by panel and board handlers (it was being done twice per frame, and
        # panel homographies mixed corners from the distorted image)
        self.undistorted_image = self.distortion_handler.undistortImage(original_image)
        self.corners, self.ids = detectAllArucos(self.undistorted_image)
        self.corners, self.ids = self.filterValidArucos(self.corners, self.ids)
        # NOTE: aruco corner smoothing (smoothArucos) was tried to stabilise the
        # homography but it degraded contour detection and lost trials; reverted.

        self.init_frame_number = min(self.init_frame_number, capture_idx)
        self.last_frame_number = capture_idx

        self.norm_coord_list = self.eye_data_handler.step(capture_idx)
        self.gaze_classification = []
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
        
        if self.norm_coord_list: self.fixation_data_store[self.current_state] += len(self.norm_coord_list)
        self.frame_data_store[self.current_state] += 1
        self.frame_data_store['total'] += 1
        self.board_metrics_now['status'] = self.current_state
        
        # key = cv.waitKey()
        # if key == ord('q') or key == ord('Q') or key == 27:
        #     exit()
        
        self.tm.stop()

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

        self.board_handler.step(undistorted_image, self.corners, self.ids)

        # Gaze can already be projected while the panel is being removed (board pose
        # from arucos + session reference grid); stored flagged as pre_start
        self.processEarlyGaze(capture_idx, desnormalized_coord_list)

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
            # but only after a warmup so the panel removal does not fake a touch
            self.board_handler.initTargetTracking(self.board_metrics_now['target_cord'])
            self.target_occlusion_counter = 0
            self.target_occlusion_start_frame = None
            self.target_warmup_end = capture_idx + self.target_warmup_frames
            self.target_tracking_active = False

        if self.is_error_init_state(undistorted_image, capture_idx, desnormalized_coord_list): return

        self.board_handler.step(undistorted_image, self.corners, self.ids)
        if self.board_handler.contour_detected_raw:
            self.last_raw_contour_frame = capture_idx
        coord_data_list = self.board_handler.getPixelInfo(desnormalized_coord_list)

        for index, coord_data in enumerate(coord_data_list):
            color, shape, slot, board_coord, corrected_coord = coord_data
            self.recordGazeSample(capture_idx, color, shape, slot, board_coord, corrected_coord, phase='execution')
            und_coord = self.distortion_handler.correctCoordinates(desnormalized_coord_list[index], homography=None)[0]
            self.gaze_classification.append((und_coord, 'execution' if color != 'not_board' else 'not_board'))

        ## TARGET TOUCH (best-effort MARK, does NOT close the trial). The trial END is
        # decided by the contour loss below (the v1.0 criterion), which keeps the exact
        # v1.0 timing. The touch watch continues into test_motor_recovery (see
        # _trackTargetTouch): the touch physically happens when the hand is already over
        # the board (contour being lost), so watching it only here missed most touches.
        self._trackTargetTouch(capture_idx)

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
            return
        # Activation needs the board clear of the panel; it can only latch while the
        # contour is still seen (in test_execution). Once active it stays active into
        # the motor phase.
        if not self.target_tracking_active:
            if capture_idx >= self.target_warmup_end and self.board_handler.contour_detected_raw \
               and self.board_handler.isTargetAreaClear(self.board_metrics_now.get('target_cord')):
                self.target_tracking_active = True
        if not self.target_tracking_active:
            return
        self.last_occlusion_measure = self.board_handler.getTargetOcclusionMeasure()
        if self.last_occlusion_measure is not None:
            frac_target, frac_control = self.last_occlusion_measure
            if frac_target > self.target_occlusion_threshold \
               and (frac_target - frac_control) > self.target_occlusion_separation:
                if self.target_occlusion_counter == 0:
                    self.target_occlusion_start_frame = capture_idx
                self.target_occlusion_counter += 1
            else:
                self.target_occlusion_counter = 0
        if self.target_occlusion_counter >= self.target_occlusion_confirm_threshold:
            self.board_metrics_now['target_touch_capture'] = self.target_occlusion_start_frame

    ## Keys of board_metrics that are not color counters
    METADATA_KEYS = ('init_capture', 'end_capture', 'early_init_capture', 'motor_onset_capture',
                     'target_touch_capture', 'hand_exit_capture', 'sequence', 'trial_id', 'status',
                     'target_cord', 'target_norm_coord', 'transition_error', 'missing_trial_error')

    """
        Drops gaze samples recorded after end_frame and rebuilds the per color/shape
        counters, so counters, sequence and duration refer to the same time span.
        Only execution-phase samples count: pre_start ones live in the sequence but
        are not part of the per color/shape summary.
    """
    def trimTrialToFrame(self, metrics, end_frame):
        if 'sequence' not in metrics:
            return
        metrics['sequence'] = [s for s in metrics['sequence'] if s['frame'] <= end_frame]
        for key in [k for k in metrics.keys() if k not in self.METADATA_KEYS]:
            del metrics[key]
        for s in metrics['sequence']:
            if s.get('phase', 'execution') != 'execution':
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
    def recordGazeSample(self, capture_idx, color, shape, slot, board_coord, corrected_coord, phase):
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

        self.board_metrics_now['sequence'].append({'color': color,
                                                   'shape': shape,
                                                   'slot': slot,
                                                   'frame': capture_idx,
                                                   'phase': phase,
                                                   'board_coord': list(board_coord),
                                                   'norm_board_coord': self.board_handler.getPixelBoardNorm(corrected_coord.tolist()).tolist()})

    """
        Gaze during the panel removal window: the board pose may be known (arucos +
        session reference grid) before the full border is visible. Samples on board
        cells are stored with phase pre_start; samples over the sample panel polygon
        or over blank white (panel still covering that area) are discarded.
    """
    def processEarlyGaze(self, capture_idx, desnormalized_coord_list):
        if not desnormalized_coord_list:
            return
        panel_polygon = self.panel_handler.getPanelPolygon()

        for coordinates in desnormalized_coord_list:
            und_coord = self.distortion_handler.correctCoordinates(coordinates, homography=None)[0]

            if panel_polygon is not None and \
               cv.pointPolygonTest(panel_polygon, (float(und_coord[0]), float(und_coord[1])), False) >= 0:
                self.gaze_classification.append((und_coord, 'on_panel'))
                continue

            coord_info = self.board_handler.getPixelInfo([coordinates])
            if not coord_info or coord_info[0][0] == 'not_board':
                self.gaze_classification.append((und_coord, 'not_board'))
                continue

            color, shape, slot, board_coord, corrected_coord = coord_info[0]
            if self.board_handler.isRegionBlank(corrected_coord):
                # Flat white where a piece or slot outline should be visible: the
                # panel (or another blank surface) is covering this area
                self.gaze_classification.append((und_coord, 'blank'))
                continue

            self.recordGazeSample(capture_idx, color, shape, slot, board_coord, corrected_coord, phase='pre_start')
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
        # processPanel applies the same 2-frame confirmation init uses, so the counter
        # carries over and init picks the panel up without re-counting from scratch.
        next_panel = self.processPanel(undistorted_image, capture_idx, desnormalized_coord_list)
        if next_panel is not None:
            self.current_state = "test_finish_execution"
            return

        self.board_handler.step(undistorted_image, self.corners, self.ids)

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
        self.motor_recovery_deadline = None
        self.motor_recovery_streak = 0
        self.motor_recovery_misses = 0
        self.motor_recovery_exit_frame = None
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

    def store_results(self, output_path, participant_id = "", video_fps = None):

        self.handle_end_of_video()
        self.board_metrics_store.pop('latest', None)

        gaze_sampling_rate = getattr(self.eye_data_handler, 'gaze_sampling_rate', None)
        data_store = {'sw_version': __version__, 'video_fps': video_fps, 'gaze_sampling_rate': gaze_sampling_rate, 'participant_id': participant_id, 'frames_info': self.frame_data_store, 'fixations_info': self.fixation_data_store, 'trials_data': self.board_metrics_store}
        with open(os.path.join(output_path,f'data_{participant_id}.pkl'), 'wb') as f:
            pickle.dump(data_store, f)
        
        dumpYaml(os.path.join(output_path,f'data_{participant_id}.yaml'), data_store, 'w')

        ## CSV trials
        csv_data = []
        csv_data_seq = []
        # Two blocks of columns: derived per-phase durations (convenient) and the raw
        # event frames in the World video (so any analysis can recompute its own
        # intervals / pick its own trial-end point). Empty when the event was not seen.
        csv_data.append(['block_index', 'trial_index', 'trial_name', 'Color', 'Shape', 'Piece Fixations', 'Slot only Fixations',
                         'trial_duration_s', 'early_start_duration_s', 'time_to_target_s', 'search_duration_s', 'motor_duration_s',
                         'frame_early_init', 'frame_init', 'frame_target_found', 'frame_motor_onset', 'frame_target_touch', 'frame_hand_exit', 'frame_end',
                         'Finish Status'])
        csv_data_seq.append(['block_index', 'trial_index', 'trial_name', 'Color', 'Shape', 'Piece=1/Slot=0', 'Phase', 'Frame_N', 'trial_duration_s', 'Board Coord', 'Board norm Coord', 'Finish Status'])
        for block_id, trial_id in sorted(self.board_metrics_store.keys()):
            trial_metric = self.board_metrics_store[(block_id, trial_id)]
            board_metrics = list(trial_metric.values())[0]
            board_test_name = list(trial_metric.keys())[0]
            if not 'init_capture' in board_metrics or not 'end_capture' in board_metrics:
                continue
            duration_s = (board_metrics['end_capture']-board_metrics['init_capture'])/self.video_fps
            init_capture = board_metrics['init_capture']
            end_capture = board_metrics['end_capture']
            # Search starts when the participant first looks at the board, which may be
            # during the panel removal (early gaze), not at the full-board start
            search_start = board_metrics.get('early_init_capture', init_capture)
            motor_onset = board_metrics.get('motor_onset_capture')
            # Time with gaze already on the board during the panel removal, before the
            # formal full-board start (pre_start phase samples)
            early_duration_s = 0
            if 'early_init_capture' in board_metrics:
                early_duration_s = max(0, (init_capture-board_metrics['early_init_capture'])/self.video_fps)
            # First gaze on the target cell (when it was first found with the eyes).
            # target_cord is [row,col]; sequence board_coord is [col,row]
            first_target_frame = None
            target_cord = board_metrics.get('target_cord')
            if target_cord and target_cord[0] is not None:
                target_colrow = [int(target_cord[1]), int(target_cord[0])]
                for step in board_metrics.get('sequence', []):
                    if list(step['board_coord']) == target_colrow:
                        first_target_frame = step['frame']
                        break
            time_to_target_s = '' if first_target_frame is None else max(0, (first_target_frame-search_start)/self.video_fps)
            # Search time (look at board until the hand enters) and motor/reach time
            # (hand enters until the touch). Empty when the motor onset was not seen.
            search_duration_s, motor_duration_s = '', ''
            if motor_onset is not None:
                search_duration_s = max(0, (motor_onset-search_start)/self.video_fps)
                motor_duration_s = max(0, (end_capture-motor_onset)/self.video_fps)

            # Cognitive phase of a given gaze frame, for the sequence CSV
            def phaseOf(frame, base_phase):
                if base_phase == 'pre_start':
                    return 'pre_start'
                if motor_onset is not None and frame >= motor_onset:
                    return 'motor'
                if first_target_frame is not None and frame >= first_target_frame:
                    return 'verification'
                return 'search'

            # Raw event frames (empty string when the event was not observed)
            f_early = board_metrics.get('early_init_capture', '')
            f_target = first_target_frame if first_target_frame is not None else ''
            f_motor = motor_onset if motor_onset is not None else ''
            f_touch = board_metrics.get('target_touch_capture', '')
            f_exit = board_metrics.get('hand_exit_capture', '')

            for color, color_item in board_metrics.items():
                if color in self.METADATA_KEYS:
                    continue
                for shape, shape_item in color_item.items():
                    csv_data.append([block_id, trial_id, board_test_name, color, shape, shape_item[True], shape_item[False],
                                     duration_s, early_duration_s, time_to_target_s, search_duration_s, motor_duration_s,
                                     f_early, init_capture, f_target, f_motor, f_touch, f_exit, end_capture,
                                     board_metrics['status']])

            for step in board_metrics['sequence']:
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
                    board_metrics['status']
                ])

        with open(os.path.join(output_path,f'trials_data_{participant_id}.csv'), mode="w", newline="") as file:
            csv.writer(file).writerows(csv_data)
        
        with open(os.path.join(output_path,f'trials_data_{participant_id}_sequence.csv'), mode="w", newline="") as file:
            csv.writer(file).writerows(csv_data_seq)

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
