#!/usr/bin/env python3
# encoding: utf-8

import copy

import cv2 as cv
import numpy as np

from src.utils import getMosaic

"""
    State Machine that handles program execution
"""
class StateMachine:
    def __init__(self, board_handler, panel_handler, eye_data_handler):
        # Estado init
        self.current_state = "init"

        self.board_handler = board_handler
        self.panel_handler = panel_handler
        self.eye_data_handler = eye_data_handler

        self.board_metrics_store = []
        self.board_metrics_now = {}
        self.current_test_key = None

        self.norm_coord = None
        self.desnormalized_coord = None


        self.board_contour_switch_state_threshold = 6
        self.board_contour_nondetected_counter = 0
        self.board_contour_detected_counter = 0

        self.tm = cv.TickMeter()

        # Speed up those part that do not need so much precision in processing
        self.frame_speed_multiplier = 1
    
    def visualization(self, original_image, capture_idx, frame_width, frame_height, participan_id = ""):
        board_view_cfg, board_view_detected = self.board_handler.getVisualization(original_image)    
        image_board_cfg, image_board_detected = self.board_handler.getUndistortedVisualization(original_image)

        panel_view = self.panel_handler.getVisualization()
        
        if self.norm_coord is not None:
            cv.circle(image_board_cfg, self.desnormalized_coord[0], radius=10, color=(0,255,0), thickness=-1)
            cv.circle(image_board_detected, self.desnormalized_coord[0], radius=10, color=(0,255,0), thickness=-1)
        
        ## Just logging stuff :)
        debug_data_view = np.zeros_like(panel_view)
        debug_data = [f"Participant: {participan_id}", f'Current state: {self.current_state}']
        if self.current_test_key is not None:
            debug_data.append(f"Current Test: search -> {self.current_test_key['color']} {self.current_test_key['shape']}")

            if 'init_capture' in self.board_metrics_now:
                debug_data.append(f"    - Started at {self.board_metrics_now['init_capture']} frame.")
        elif len(self.board_metrics_store) > 1:
            board_metrucs_prev = self.board_metrics_store[-1]
            debug_data.append(f"Previous Test: search -> {board_metrucs_prev['color']} {board_metrucs_prev['shape']}")

            if 'init_capture' in board_metrucs_prev:
                debug_data.append(f"    - Started at {board_metrucs_prev['init_capture']} frame.")
            if 'end_capture' in board_metrucs_prev:
                debug_data.append(f"    - Ended at {board_metrucs_prev['end_capture']} frame.")
        
        mosaic = getMosaic(capture_idx, frame_width, frame_height, titles_list=['Complete Cfg', 'Board Cfg', 'Panel View', 'Complete Detected', 'Board Detected', 'Debug'], 
                     images_list=[image_board_cfg, board_view_cfg, panel_view, image_board_detected, board_view_detected, debug_data_view], 
                     debug_data_list=[None,None,None,None,None,debug_data],
                     rows=2, cols=3, resize = 2/5)
        
        cv.putText(mosaic, f"FPS: {self.tm.getFPS():.1f} ", org=(3, 10),
                fontFace=cv.FONT_HERSHEY_DUPLEX, fontScale=0.3, color=(0,0,0), thickness=1, lineType=cv.LINE_AA)
    
        return mosaic, cv.resize(original_image, (frame_width, frame_height))
        
    def processPanel(self, original_image, capture_idx, desnormalized_coord):
        self.panel_handler.step(original_image)

        shape, aruco, panel = self.panel_handler.getPixelInfo(desnormalized_coord)
        current_panel = self.panel_handler.getCurrentPanel()

        return current_panel

    def getFrameMultiplier(self):
        return min(1, self.frame_speed_multiplier)

    """
        State machine control loop
    """
    def step(self, original_image, capture_idx):
        self.tm.start()
        
        self.norm_coord = self.eye_data_handler.step(capture_idx)
        self.desnormalized_coord = None
        if self.norm_coord is not None:
            # print(f'{capture_idx =}')
            desnormalized_x = int(self.norm_coord[0] * original_image.shape[1])
            desnormalized_y = int(self.norm_coord[1] * original_image.shape[0])
            self.desnormalized_coord = np.array([[desnormalized_x, desnormalized_y]])

        # print(f'{norm_coord = }')
        # print(f'{desnormalized_coord = }')


        if self.current_state == "init":
            self.init_state(original_image, capture_idx, self.desnormalized_coord)
            self.frame_speed_multiplier = 50
        elif self.current_state == "get_test_name":
            self.get_test_name_state(original_image, capture_idx, self.desnormalized_coord)
            self.frame_speed_multiplier = 15
        elif self.current_state == "test_start_execution":
            self.test_start_execution_state(original_image, capture_idx, self.desnormalized_coord)
            self.frame_speed_multiplier = 1
        elif self.current_state == "test_execution":
            self.test_execution_state(original_image, capture_idx, self.desnormalized_coord)
            self.frame_speed_multiplier = 1
        elif self.current_state == "test_finish_execution":
            self.test_finish_execution_state(original_image, capture_idx, self.desnormalized_coord)
            self.frame_speed_multiplier = 1

        self.tm.stop()

    def init_state(self, original_image, capture_idx, desnormalized_coord):
        
        current_panel = self.processPanel(original_image, capture_idx, desnormalized_coord)
        if current_panel is not None:
            self.current_test_key = current_panel
            self.current_state = "get_test_name"
            print(f"[StateMachine::init_state] Switch to get_test_name state. Test panel detected.")

    def get_test_name_state(self, original_image, capture_idx, desnormalized_coord):
        
        current_panel = self.processPanel(original_image, capture_idx, desnormalized_coord)
        if current_panel is None:
            self.current_state = "test_start_execution"
            print(f"[StateMachine::get_test_name] Switch to test_start_execution. Gathering data for test {self.current_test_key}")

    def test_start_execution_state(self, original_image, capture_idx, desnormalized_coord):
        
        self.board_handler.step(original_image)
        
        if not self.board_handler.isContourDetected():
            self.board_contour_detected_counter += 1
        else:
            self.board_contour_detected_counter = 0

        # Wait until contour of board can be fully detected 
        if self.board_contour_detected_counter > self.board_contour_switch_state_threshold:
            self.board_contour_detected_counter = 0
            self.current_state = "test_execution"
            print(f"[StateMachine::test_start_execution] Switch to test_execution. Gathering data for test {self.current_test_key}")


    def test_execution_state(self, original_image, capture_idx, desnormalized_coord):

        if not 'init_capture' in self.board_metrics_now:
            self.board_metrics_now['init_capture'] = capture_idx

        self.board_handler.step(original_image)
        color, shape, slot, board_coord = self.board_handler.getPixelInfo(desnormalized_coord)

        # Update board metrics
        if color is not None:
            if color not in self.board_metrics_now:
                self.board_metrics_now[color] = {shape: {True: 0, False: 0}}
            if shape not in self.board_metrics_now[color]:
                self.board_metrics_now[color][shape] = {True: 0, False: 0}

            self.board_metrics_now[color][shape][slot] += 1

        if not self.board_handler.isContourDetected():
            self.board_contour_nondetected_counter += 1
        else:
            self.board_contour_nondetected_counter = 0

        # key = cv.pollKey()
        # if key == ord('f') or key == ord('f'):
        if self.board_contour_nondetected_counter > self.board_contour_switch_state_threshold:
            self.board_contour_nondetected_counter = 0
            self.current_state = "test_finish_execution"
            print(f"[StateMachine::test_execution] Switch to test_finish_execution. Waiting fo new test to start.")

    def test_finish_execution_state(self, original_image, capture_idx, desnormalized_coord):
        
        self.board_metrics_now['end_capture'] = capture_idx
        self.board_metrics_store.append({f'{self.current_test_key}': copy.deepcopy(self.board_metrics_now)})
        self.board_metrics_now = {}
        self.current_test_key = None

        self.current_state = "init"
        print(f"[StateMachine::test_finish_execution] Switch to init. Gathering data for test {self.current_test_key}")


      

    def print_results(self, video_fps):
        print(self.board_metrics_store)

        return 
        for color, shapes_dict in self.board_metrics_store.items():
            time_color = 0
            for shape, slot_dict in shapes_dict.items():
                for slot, num in slot_dict.items():
                    num = float(num)/float(video_fps)
                    time_color += num

            print(f'------------------------------')
            print(f'Color: {color}: {time_color}s')
            for shape, slot_dict in shapes_dict.items():
                for slot, num in slot_dict.items():
                    print(f'Shape: {shape} ({slot}): {num}s')