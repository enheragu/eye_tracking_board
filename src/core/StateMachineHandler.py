#!/usr/bin/env python3
# encoding: utf-8
import os
import copy

import math
import cv2 as cv
import numpy as np

import csv
import pickle
import yaml
from yaml.loader import SafeLoader
from tabulate import tabulate

from src.core.utils import dumpYaml, parseYaml
from src.core.utils import bcolors
from src.core.utils import log
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

        self.tm = cv.TickMeter()

        # Speed up those part that do not need so much precision in processing
        self.frame_speed_multiplier = 1

        self.state_info = {
            "init": {'callback': self.init_state, 'frame_mult': 1 if slow_analysis else int(self.video_fps*0.25)},
            "get_test_name": {'callback': self.get_test_name_state, 'frame_mult': 1 if slow_analysis else int(self.video_fps*0.25)},
            "test_start_execution": {'callback': self.test_start_execution_state, 'frame_mult': 1},
            "test_execution": {'callback': self.test_execution_state, 'frame_mult': 1},
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

        ## Gaze samples over the camera view. Gaze coords live in the original
        # (distorted) frame, project them to the undistorted view being displayed
        if self.norm_coord_list and self.board_handler.display_fixation:
            for desnormalized_coord in self.desnormalized_coord_list:
                if desnormalized_coord[0][0] < 0 or desnormalized_coord[0][1] < 0:
                    continue
                und_coord = self.distortion_handler.correctCoordinates(desnormalized_coord, homography=None)
                und_coord = (int(und_coord[0][0]), int(und_coord[0][1]))
                cv.circle(canvas, und_coord, radius=5, color=(0,255,0), thickness=-1)

        ## Status panel (top-left)
        debug_data = [f"Participant: {participan_id}   Frame: {capture_idx}/{last_capture_idx}   FPS: {self.tm.getFPS():.1f}",
                      f"State: {self.current_state}"]
        if self.current_test_key is not None:
            debug_data.append(f"Current Test: Block: {self.test_block_count}, trial: {self.test_trial_count-1};  trial_id: {self.trial_id}")
            debug_data.append(f"   Search -> {self.current_test_key['color']} {self.current_test_key['shape']} (arucos: {self.current_test_key['arucos']})")
            if 'init_capture' in self.board_metrics_now:
                debug_data.append(f"   Started at frame {self.board_metrics_now['init_capture']}")
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

        self.init_frame_number = min(self.init_frame_number, capture_idx)
        self.last_frame_number = capture_idx

        self.norm_coord_list = self.eye_data_handler.step(capture_idx)
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

    ## Used in case something happends and some state change is missed. If a new pannel
    # appears while a trial is running with data already gathered, the trial is closed
    # as valid (the board stopped being attended; end is backdated to its last real
    # sighting) with a distinctive status. Without gathered data it is stored as error.
    def is_error_init_state(self, undistorted_image, capture_idx, desnormalized_coord_list):
        current_panel = self.processPanel(undistorted_image, capture_idx, desnormalized_coord_list)
        if current_panel is not None and \
            not IsSamePanel(current_panel, self.current_test_key):

            if 'init_capture' in self.board_metrics_now:
                # Trial was running: the hand occlusion lost the race against the next
                # panel detection. Close it at the last frame the board was visible.
                end_capture = self.last_raw_contour_frame if self.last_raw_contour_frame is not None else capture_idx
                self.board_metrics_now['end_capture'] = end_capture
                self.trimTrialToFrame(self.board_metrics_now, end_capture)
                self.board_metrics_now['status'] = 'test_finish_by_next_panel'
                self.board_metrics_now['trial_id'] = self.trial_id
                key = f"{self.current_test_key['color']}_{self.current_test_key['shape']}"
                logStateChange(f"[StateMachine::error_init_state] [{capture_idx}] New panel detected ({current_panel}) while trial for {self.current_test_key} was running. Trial closed at frame {end_capture} with status test_finish_by_next_panel.")
            else:
                self.board_metrics_now['transition_error'] = {'transition_error': {True: 0, False: 0}}
                self.board_metrics_now['end_capture'] = capture_idx
                self.board_metrics_now['status'] = self.current_state
                self.board_metrics_now['trial_id'] = self.trial_id
                key = f"transition_error_no_init_{self.current_test_key['color']}_{self.current_test_key['shape']}"
                logErrorMsg(f"[StateMachine::error_init_state] [{capture_idx}] ERROR IN TRANSITION. New panel detected ({current_panel}), previous panel was {self.current_test_key}. Switch to get_test_name state. Test panel detected.")

            self.board_metrics_store[(self.block_id, self.trial_id)] = {key: copy.deepcopy(self.board_metrics_now)}
            self.board_metrics_store['latest'] = self.board_metrics_store[(self.block_id, self.trial_id)]
            self.board_metrics_now = {}
            self.current_test_key = None
            self.last_raw_contour_frame = None

            self.current_state = "init"
            return True
        return False
    
    def init_state(self, undistorted_image, capture_idx, desnormalized_coord_list):
        current_panel = self.processPanel(undistorted_image, capture_idx, desnormalized_coord_list)
        if current_panel is not None:
            self.current_test_key = current_panel
            detected_trial = f"{self.current_test_key['color']}_{self.current_test_key['shape']}"

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

    def test_start_execution_state(self, undistorted_image, capture_idx, desnormalized_coord_list):
        if self.is_error_init_state(undistorted_image, capture_idx, desnormalized_coord_list): return

        self.board_handler.step(undistorted_image, self.corners, self.ids)

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
            self.board_metrics_now['sequence'] = []
            self.last_raw_contour_frame = capture_idx

        if self.is_error_init_state(undistorted_image, capture_idx, desnormalized_coord_list): return

        self.board_handler.step(undistorted_image, self.corners, self.ids)
        if self.board_handler.contour_detected_raw:
            self.last_raw_contour_frame = capture_idx
        coord_data_list = self.board_handler.getPixelInfo(desnormalized_coord_list)

        for coord_data in coord_data_list:
            color, shape, slot, board_coord, corrected_coord  = coord_data
            # Update board metrics
            if color not in self.board_metrics_now:
                self.board_metrics_now[color] = {shape: {True: 0, False: 0}}
            if shape not in self.board_metrics_now[color]:
                self.board_metrics_now[color][shape] = {True: 0, False: 0}

            self.board_metrics_now[color][shape][slot] += 1
            self.board_metrics_now['sequence'].append({'color':color,
                                                       'shape':shape,
                                                       'slot':slot, 
                                                       'frame':capture_idx, 
                                                       'board_coord':list(board_coord),
                                                       'norm_board_coord': self.board_handler.getPixelBoardNorm(corrected_coord.tolist()).tolist()})

        if not self.board_handler.isContourDetected():
            self.board_contour_nondetected_counter += 1
        else:
            self.board_contour_nondetected_counter = 0

        # key = cv.pollKey()
        # if key == ord('f') or key == ord('f'):
        if self.board_contour_nondetected_counter > self.board_contour_switch_state_threshold:
            self.board_contour_nondetected_counter = 0
            self.current_state = "test_finish_execution"
            logStateChange(f"[StateMachine::test_execution] [{capture_idx}] Switch to test_finish_execution. Waiting fo new test to start.")
            
    ## Keys of board_metrics that are not color counters
    METADATA_KEYS = ('init_capture', 'end_capture', 'sequence', 'trial_id', 'status',
                     'target_cord', 'target_norm_coord', 'transition_error', 'missing_trial_error')

    """
        Drops gaze samples recorded after end_frame and rebuilds the per color/shape
        counters, so counters, sequence and duration refer to the same time span.
    """
    def trimTrialToFrame(self, metrics, end_frame):
        if 'sequence' not in metrics:
            return
        metrics['sequence'] = [s for s in metrics['sequence'] if s['frame'] <= end_frame]
        for key in [k for k in metrics.keys() if k not in self.METADATA_KEYS]:
            del metrics[key]
        for s in metrics['sequence']:
            if s['color'] not in metrics:
                metrics[s['color']] = {s['shape']: {True: 0, False: 0}}
            if s['shape'] not in metrics[s['color']]:
                metrics[s['color']][s['shape']] = {True: 0, False: 0}
            metrics[s['color']][s['shape']][s['slot']] += 1

    def test_finish_execution_state(self, undistorted_image, capture_idx, desnormalized_coord_list):

        # Backdate trial end to the last frame the board was actually visible:
        # the confirmation counter plus contour inertia otherwise add a systematic
        # ~5-7 frames (~0.2s) of occlusion time to every trial duration
        end_capture = self.last_raw_contour_frame if self.last_raw_contour_frame is not None else capture_idx
        self.board_metrics_now['end_capture'] = end_capture
        self.trimTrialToFrame(self.board_metrics_now, end_capture)
        self.board_metrics_now['status'] = self.current_state
        self.board_metrics_store[(self.block_id, self.trial_id)] = {f"{self.current_test_key['color']}_{self.current_test_key['shape']}": copy.deepcopy(self.board_metrics_now)}
        self.board_metrics_store['latest'] = self.board_metrics_store[(self.block_id, self.trial_id)]
        self.board_metrics_now = {}
        self.current_test_key = None
        self.last_raw_contour_frame = None

        self.current_state = "init"
        logStateChange(f"[StateMachine::test_finish_execution::] Switch to init. Trial end backdated to frame {end_capture} (confirmed at {capture_idx}).")

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
            self.board_metrics_now = {}
            self.current_test_key = None

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

        data_store = {'sw_version': __version__, 'video_fps': video_fps, 'participant_id': participant_id, 'frames_info': self.frame_data_store, 'fixations_info': self.fixation_data_store, 'trials_data': self.board_metrics_store}
        with open(os.path.join(output_path,f'data_{participant_id}.pkl'), 'wb') as f:
            pickle.dump(data_store, f)
        
        dumpYaml(os.path.join(output_path,f'data_{participant_id}.yaml'), data_store, 'w')

        ## CSV trials
        csv_data = []
        csv_data_seq = []
        csv_data.append(['block_index', 'trial_index', 'trial_name', 'Color', 'Shape', 'Piece Fixations', 'Slot only Fixations', 'trial_duration_s', 'Finish Status'])
        csv_data_seq.append(['block_index', 'trial_index', 'trial_name', 'Color', 'Shape', 'Piece=1/Slot=0', 'Frame_N', 'trial_duration_s', 'Board Coord', 'Board norm Coord', 'Finish Status'])
        for block_id, trial_id in sorted(self.board_metrics_store.keys()):
            trial_metric = self.board_metrics_store[(block_id, trial_id)]
            board_metrics = list(trial_metric.values())[0]
            board_test_name = list(trial_metric.keys())[0]
            if not 'init_capture' in board_metrics or not 'end_capture' in board_metrics:
                continue
            duration_s = (board_metrics['end_capture']-board_metrics['init_capture'])/self.video_fps
            
            for color, color_item in board_metrics.items():
                if color in ['init_capture', 'end_capture', 'sequence', 'trial_id', 'status', 'target_cord', 'target_norm_coord']:
                    continue
                for shape, shape_item in color_item.items():
                    csv_data.append([block_id, trial_id, board_test_name, color, shape, shape_item[True], shape_item[False], duration_s, board_metrics['status']])
            
            for step in board_metrics['sequence']:
                csv_data_seq.append([
                    block_id,
                    trial_id,
                    board_test_name,
                    step['color'],
                    step['shape'],
                    step['slot'],
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
                if color in ['init_capture', 'end_capture', 'sequence', 'trial_id', 'status', 'target_cord', 'target_norm_coord']:
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
