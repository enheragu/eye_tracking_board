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
from tabulate import tabulate

from src.utils import getMosaic, bcolors


def printStateChange(msg):
    print(f"{bcolors.BOLD}{bcolors.OKCYAN}{msg}{bcolors.ENDC}")
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


        self.board_contour_switch_state_threshold = 4
        self.board_contour_nondetected_counter = 0
        self.board_contour_detected_counter = 0

        self.tm = cv.TickMeter()

        # Speed up those part that do not need so much precision in processing
        self.frame_speed_multiplier = 1


        self.state_info = {
            "init": {'callback': self.init_state, 'frame_mult': 1},
            "get_test_name": {'callback': self.get_test_name_state, 'frame_mult': 1},
            "test_start_execution": {'callback': self.test_start_execution_state, 'frame_mult': 1},
            "test_execution": {'callback': self.test_execution_state, 'frame_mult': 1},
            "test_finish_execution": {'callback': self.test_finish_execution_state, 'frame_mult': 1} 
        }

        state_keys = list(self.state_info.keys())
        # Storage for fixation analysis on each state
        self.fixation_data_store = dict.fromkeys(state_keys+['total'], 0)
        # Storage for frame analysis on each state
        self.frame_data_store = dict.fromkeys(state_keys+['total'], 0)


        self.init_frame_number = math.inf
        self.last_frame_number = math.inf
    
    def visualization(self, original_image, capture_idx, last_capture_idx, frame_width, frame_height, participan_id = ""):
        board_view_cfg, board_view_detected = self.board_handler.getVisualization(original_image)    
        image_board_cfg, image_board_detected = self.board_handler.getDistortedOriginalVisualization(original_image)

        panel_view = self.panel_handler.getVisualization()
        
        if self.norm_coord is not None:
            cv.circle(image_board_cfg, self.desnormalized_coord[0], radius=5, color=(0,255,0), thickness=-1)
            cv.circle(image_board_detected, self.desnormalized_coord[0], radius=5, color=(0,255,0), thickness=-1)
        
        ## Just logging stuff :)
        debug_data_view = np.zeros_like(panel_view)
        debug_data = [f"Participant: {participan_id}", f'Current state: {self.current_state}']
        if self.current_test_key is not None:
            debug_data.append(f"Current Test: search -> {self.current_test_key['color']} {self.current_test_key['shape']}")

            if 'init_capture' in self.board_metrics_now:
                debug_data.append(f"    - Started at {self.board_metrics_now['init_capture']} frame.")
        elif len(self.board_metrics_store) > 0:
            board_metrics_prev = list(self.board_metrics_store[-1].values())[0]
            board_metrics_prev_test = list(self.board_metrics_store[-1].keys())[0]
            debug_data.append(f"Previous Test: search -> {board_metrics_prev_test}")

            if 'init_capture' in board_metrics_prev:
                debug_data.append(f"    - Started at {board_metrics_prev['init_capture']} frame.")
            if 'end_capture' in board_metrics_prev:
                debug_data.append(f"    - Ended at {board_metrics_prev['end_capture']} frame.")
                debug_data.append(f"    - Took {board_metrics_prev['end_capture']-board_metrics_prev['init_capture']} frames.")
        
        mosaic = getMosaic(capture_idx=capture_idx, last_capture_idx=last_capture_idx, fps=self.tm.getFPS(),
                           frame_width=frame_width, frame_height=frame_height, 
                           titles_list=['Complete Cfg', 'Board Cfg', 'Panel View', 'Complete Detected', 'Board Detected', 'Debug'], 
                           images_list=[image_board_cfg, board_view_cfg, panel_view, image_board_detected, board_view_detected, debug_data_view], 
                           debug_data_list=[None,None,None,None,None,debug_data],
                           rows=2, cols=3, resize = 2/5)
        
        # mosaic_store = getMosaic(capture_idx, frame_width, frame_height, titles_list=['Board Detected', 'Debug'], 
        #              images_list=[board_view_detected, debug_data_view], 
        #              debug_data_list=[None,debug_data],
        #              rows=2, cols=1, resize = 2/5)
        

        return mosaic, cv.resize(mosaic, (frame_width*3, frame_height*2))
        
    def processPanel(self, original_image, capture_idx, desnormalized_coord):
        self.panel_handler.step(original_image)

        shape, aruco, panel = self.panel_handler.getPixelInfo(desnormalized_coord)
        current_panel = self.panel_handler.getCurrentPanel()

        return current_panel

    def getFrameMultiplier(self):
        return max(1, self.frame_speed_multiplier)

    """
        State machine control loop
    """
    def step(self, original_image, capture_idx):
        self.tm.start()

        self.init_frame_number = min(self.init_frame_number, capture_idx)
        self.last_frame_number = capture_idx
        
        self.norm_coord = self.eye_data_handler.step(capture_idx)
        self.desnormalized_coord = None
        if self.norm_coord is not None:
            # print(f'{capture_idx =}')
            desnormalized_x = int(self.norm_coord[0] * original_image.shape[1])
            desnormalized_y = int(self.norm_coord[1] * original_image.shape[0])
            self.desnormalized_coord = np.array([[desnormalized_x, desnormalized_y]])
            self.fixation_data_store['total'] += 1
        # print(f'{norm_coord = }')
        # print(f'{desnormalized_coord = }')

        self.state_info[self.current_state]['callback'](original_image, capture_idx, self.desnormalized_coord)
        self.frame_speed_multiplier = self.state_info[self.current_state]['frame_mult']
        
        if self.norm_coord is not None: self.fixation_data_store[self.current_state] += 1
        self.frame_data_store[self.current_state] += 1
        self.frame_data_store['total'] += 1
        # key = cv.waitKey()
        # if key == ord('q') or key == ord('Q') or key == 27:
        #     exit()

        self.tm.stop()

    def init_state(self, original_image, capture_idx, desnormalized_coord):
        
        current_panel = self.processPanel(original_image, capture_idx, desnormalized_coord)
        if current_panel is not None:
            self.current_test_key = current_panel
            self.current_state = "get_test_name"
            printStateChange(f"[StateMachine::init_state] [{capture_idx}] Switch to get_test_name state. Test panel detected.")

    def get_test_name_state(self, original_image, capture_idx, desnormalized_coord):
        
        current_panel = self.processPanel(original_image, capture_idx, desnormalized_coord)
        if current_panel is None:
            self.current_state = "test_start_execution"
            printStateChange(f"[StateMachine::get_test_name] [{capture_idx}] Switch to test_start_execution. Gathering data for test {self.current_test_key['shape']} {self.current_test_key['color']}")

    def test_start_execution_state(self, original_image, capture_idx, desnormalized_coord):
        
        self.board_handler.step(original_image)
        
        # if self.board_handler.isContourDetected():
        #     self.board_contour_detected_counter += 1
        # else:
        #     self.board_contour_detected_counter = 0

        # # Wait until contour of board can be fully detected 
        # if self.board_contour_detected_counter > self.board_contour_switch_state_threshold:
        if self.board_handler.isContourDetected():
            self.board_contour_detected_counter = 0
            self.current_state = "test_execution"
            printStateChange(f"[StateMachine::test_start_execution] [{capture_idx}] Switch to test_execution. Gathering data for test {self.current_test_key['shape']} {self.current_test_key['color']}")
            ## Wanna check with this current frame already
            self.test_execution_state(original_image, capture_idx, desnormalized_coord)


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
            printStateChange(f"[StateMachine::test_execution] [{capture_idx}] Switch to test_finish_execution. Waiting fo new test to start.")
            ## Wanna check with this current frame already
            self.test_finish_execution_state(original_image, capture_idx, desnormalized_coord)

    def test_finish_execution_state(self, original_image, capture_idx, desnormalized_coord):
        
        self.board_metrics_now['end_capture'] = capture_idx
        self.board_metrics_store.append({f"{self.current_test_key['color']}_{self.current_test_key['shape']}": copy.deepcopy(self.board_metrics_now)})
        self.board_metrics_now = {}
        self.current_test_key = None

        self.current_state = "init"
        printStateChange(f"[StateMachine::test_finish_execution::] Switch to init.")


    def log_results(self, video_fps, output_path):
        
        data_store = {'frames_info': self.frame_data_store, 'fixations_info': self.fixation_data_store, 'trials_data': self.board_metrics_store}
        with open(os.path.join(output_path,'data.pkl'), 'wb') as f:
            pickle.dump(data_store, f)
            
        with open(os.path.join(output_path,'data.yaml'), 'w') as file:
            yaml.dump(data_store, file, default_flow_style=False)

        ## CSV trials
        csv_data = []
        csv_data.append(['trial_index','trial_name', 'Color', 'Shape', 'Slot Fixations', 'Distractor Fixations', 'trial_duration_s'])
        for index, test_metric in enumerate(self.board_metrics_store):
            board_metrics = list(test_metric.values())[0]
            board_test_name = list(test_metric.keys())[0]
            duration_s = (board_metrics['end_capture']-board_metrics['init_capture'])/video_fps
            for color, color_item in board_metrics.items():
                if color in ['init_capture', 'end_capture']:
                    continue
                for shape, shape_item in color_item.items():
                    csv_data.append([index, board_test_name, color, shape, shape_item[True], shape_item[False], duration_s])

            
        with open(os.path.join(output_path,'trials_data.csv'), mode="w", newline="") as file:
            csv.writer(file).writerows(csv_data)

        self.print_results(video_fps=video_fps)


    def print_results(self, video_fps):
        
        ## Prints fixation data :)
        total_fixations = self.fixation_data_store['total']
        frames_with_fixation = (total_fixations/self.frame_data_store['total'])*100
        
        printStateChange(f"#############################")
        printStateChange(f"##      Result report      ##")
        printStateChange(f"#############################\n")

        ## Prints table with frame distribution
        total_frames = self.frame_data_store['total']
        printStateChange(f"Frames distribution of the {total_frames} fixations involved.")
        print(f"* Please note speed multiplier")
        print(f"\t· Analyzed from frame {self.init_frame_number} to {self.last_frame_number}")
        print(f"\t· Total frames: {self.frame_data_store['total']}")
        print(f"\t· Total time (s): {self.frame_data_store['total']/video_fps}")
        log_table_data = []
        log_table_headers = ['State Name', 'N Frames', 'Percent.', 'Time (s)']
        for key, item in self.frame_data_store.items():
            if key in ['total']:
                continue
            log_table_data.append([key,item,f"{item/total_frames*100:3f}", item/video_fps])
        
        print(tabulate(log_table_data, headers=log_table_headers, tablefmt="pretty"))
        print("\n\n")


        ## Prints table with fixation distribution
        printStateChange(f"Fixation distribution of the {total_fixations} fixations involved. Frames with fixation data: {frames_with_fixation:3f}%")
        print(f"* Please note speed multiplier")
        log_table_data = []
        log_table_headers = ['State Name', 'Fixation frames', 'Percent.']
        for key, item in self.fixation_data_store.items():
            if key in ['total']:
                continue
            log_table_data.append([key,item,f"{item/total_fixations*100:3f}"])
        
        print(tabulate(log_table_data, headers=log_table_headers, tablefmt="pretty"))
        print("\n\n")


        ## Prints data of each trial
        for index, test_metric in enumerate(self.board_metrics_store):
            board_metrics = list(test_metric.values())[0]
            test_tag = f"[Trial {index}] Search for {list(test_metric.keys())[0]} "

            log_table_headers = ['Color', 'Shape', 'Slot Fixations', 'Distractor Fixations']
            log_table_data = []
            for color, color_item in board_metrics.items():
                if color in ['init_capture', 'end_capture']:
                    continue
                for shape, shape_item in color_item.items():
                    log_table_data.append([color, shape, shape_item[True], shape_item[False]])

            formatted_table = tabulate(log_table_data, headers=log_table_headers, tablefmt="pretty")
            
            table_width = len(formatted_table.splitlines()[1]) # Get length from dashes, which is second one
            title_dashes = '-' * ((table_width - len(test_tag)) // 2)

            duration_s = (board_metrics['end_capture']-board_metrics['init_capture'])/video_fps
            printStateChange(f"{title_dashes}{test_tag}{title_dashes}")
            print(f"    - Started at {board_metrics['init_capture']} frame.")
            print(f"    - Ended at {board_metrics['end_capture']} frame.")
            print(f"    - Took {board_metrics['end_capture']-board_metrics['init_capture']} frames. ({duration_s} s)")
            for line in formatted_table.splitlines():
                print(line)
            print("\n\n")
        
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