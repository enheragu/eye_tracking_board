
import os

import time
import bisect

import threading
import numpy as np
import pandas as pd

import pickle  

from src.deps.file_methods import load_pldata_file


def check_duplicated_timestamps(data):
    timestamps_checked = set()
    timestamps_duplicated = set()

    for dic in data:
        timestamp = dic['timestamp']
        if timestamp in timestamps_checked:
            timestamps_duplicated.add(timestamp)
        else:
            timestamps_checked.add(timestamp)

    return timestamps_duplicated


"""
    When projecting fixation data take into account that theres a big discrepancy
    in how the matching occurs, theres a lot more thata in fixations that in frames, 
    so some frames can end up with more than one fixation associated:
        FPS of eye1.mp4 is 123.88
        FPS of world.mp4 is 29.81
"""
class EyeDataHandlerPLDATA:
    def __init__(self, root_path, data_path, video_fps, topic_data='fixations'):
        start_time = time.time()
        
        pldata = load_pldata_file(directory=data_path, topic=topic_data, track_progress_in_console=True)
        # print(f'{type(data.data)} - {data.data}')

        self.world_timestamps = np.load(os.path.join(root_path,'world_timestamps.npy'))
        # print(f'{self.world_timestamps = }')
        # print(f'{len(self.world_timestamps) = }')

        print(f"[EyeDataHandlerPLDATA(::__init__] Process all {topic_data} topic data to match world timestamps, fixations data from: {data_path}")
        
        self.video_fps = video_fps
        self.fixation_start_world_frame = {}
        self.data = []

        last_world_index = 0
        for index, item in enumerate(pldata.data):
            dict_obj = dict(item)
            current_timestamp = dict_obj['timestamp']
            duration = 0 if not 'duration' in dict_obj else dict_obj['duration']
            self.data.append({'norm_pos': dict_obj['norm_pos'],
                            'timestamp': current_timestamp,
                            'duration': duration})
                                
        self.data.sort(key=lambda x: x['timestamp'])
        self.world_timestamps = sorted(self.world_timestamps)
        print(f"[EyeDataHandlerPLDATA(::__init__] Total number of {topic_data}: {len(self.data)}")
        print(f"[EyeDataHandlerPLDATA(::__init__] Duplicated timestamps in {topic_data} file: {len(check_duplicated_timestamps(self.data))}")

        duplicated = 0

        for index, item in enumerate(self.data):
            timestamp = item['timestamp']
            duration = item['duration']
            duration_frames = int(self.video_fps*(duration/1000.0))
            duration_frames = 1 if duration_frames < 1 else duration_frames
            
            # Find index in which this new timestamp would 'fit'
            video_frame = bisect.bisect_right(self.world_timestamps, timestamp)

            def check_video_frame(video_frame, world_timestamps, fixation_start_world_frame, recursion = 0, max_recursion = 2):
                if video_frame >= len(world_timestamps):
                    # Should not be bigger than world_timestamps
                    video_frame = len(world_timestamps) - 1
                else:
                    video_frame = video_frame - 1
                
                video_frame = max(0, video_frame)
                
                # okey, its repeated, just assign it to next frame...
                # if recursion < max_recursion and video_frame in fixation_start_world_frame:
                #     video_frame += 1
                #     check_video_frame(video_frame, world_timestamps, fixation_start_world_frame, recursion+1)
                    
                return video_frame
            
            video_frame = check_video_frame(video_frame=video_frame, world_timestamps=self.world_timestamps, 
                                            fixation_start_world_frame=self.fixation_start_world_frame,
                                            max_recursion=1)
            # if video_frame in self.fixation_start_world_frame:
            #     duplicated += 1
            
            # Propagate duration :)
            for frame in range(video_frame, video_frame + duration_frames):
                if frame not in self.fixation_start_world_frame:
                    self.fixation_start_world_frame[frame] = []    
                
                self.fixation_start_world_frame[frame].append(index)
            # self.fixation_start_world_frame[video_frame] = index

        # print(f"[EyeDataHandlerPLDATA(::__init__] Duplicated timestamps when matching {topic_data} to world video frames: {duplicated}")
                 
        execution_time = time.time() - start_time
        
        max_index = max(list(self.fixation_start_world_frame.keys()))
        min_index = min(list(self.fixation_start_world_frame.keys()))
        
        frames_with_fixations = sum([len(fixation_list) for fixation_list in self.fixation_start_world_frame.values()])
        frame_with_max_fixations = max([len(fixation_list) for fixation_list in self.fixation_start_world_frame.values()])
        print(f"[EyeDataHandlerPLDATA(::__init__] Total number of {topic_data} data, once propagated and filtered: {frames_with_fixations}; frame with max {topic_data} data has: {frame_with_max_fixations} items") #; {self.fixation_start_world_frame[min_index] = }; {self.fixation_start_world_frame[max_index] = }"")
        print(f"[EyeDataHandlerPLDATA(::__init__] Number of frames: {len(self.world_timestamps)}; Pupil timestamp item[0] = {self.world_timestamps[0]}; Pupil timestamp item[-1] ={self.world_timestamps[-1]}")
        print(f"[EyeDataHandlerPLDATA(::__init__] Finished process, took {execution_time:.2f} seconds")

    def step(self, frame_index):
        coord_list = []
        if frame_index in self.fixation_start_world_frame:
            timestamp_idx_list = self.fixation_start_world_frame[frame_index]
            # print(f"[EyeDataHandlerPLDATA(::step] Frame {frame_index} contains {len(timestamp_idx_list)} fixation data")
            for fixation_timestamp_idx in timestamp_idx_list:
                X,Y = self.data[fixation_timestamp_idx]['norm_pos'][0], self.data[fixation_timestamp_idx]['norm_pos'][1]
                ## Flip fixation points from original coord system (bottom-left) to image
                ## coordinate system, origint at top-left
                coord_list.append([X, 1-Y])
                # print(f"[EyeDataHandlerPLDATA(::step] Data in frame {frame_index} is: {self.data[fixation_timestamp_idx]['norm_pos']}")
            
        return coord_list
    


class EyeDataHandlerCSV:
    
    def __init__(self, root_path, data_path, video_fps, topic_data='fixations'):
        start_time = time.time()

        self.video_fps = video_fps
        csv_path = os.path.join(data_path, f"{topic_data}.csv")
        self.data = pd.read_csv(csv_path)
        # print(self.data)
        
        print(f"[EyeDataHandlerCSV::__init__] Process all {topic_data} topic data to match world timestamps, fixations data from: {data_path}")
        
        
        print(f"[EyeDataHandlerCSV::__init__] Total number of {topic_data}: {self.data.shape[0]}")
           
        execution_time = time.time() - start_time        
        
        print(f"[EyeDataHandlerCSV::__init__] Finished process, took {execution_time:.2f} seconds")

    def step(self, frame_index):
        coordinate_list = []
        gaze_points = self.data[self.data["start_frame_index"] == frame_index]
        if not gaze_points.empty:
            gaze_points = gaze_points.sort_values(by="id")
            gaze_points = gaze_points[["norm_pos_x", "norm_pos_y"]]
            gaze_points = gaze_points.to_numpy()
            X, Y = gaze_points[:, 0], gaze_points[:, 1]

            # Flip the fixation points from the original coordinate system,
            # where the origin is at botton left, to the image coordinate system,
            # where the origin is at top left
            Y = 1 - Y
            coordinate_list = gaze_points.tolist()
        
        return coordinate_list