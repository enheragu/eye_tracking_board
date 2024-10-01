
import os

import time

import threading
import numpy as np

import pickle  

from src.deps.file_methods import load_pldata_file


class EyeDataHandler:
    def __init__(self, path, topic_data='fixations'):
        start_time = time.time()
        cache_file_path = os.path.join(path, 'fixations_processed.pkl')
        
        if not os.path.exists(cache_file_path):
        
            pldata = load_pldata_file(directory=path, topic=topic_data, track_progress_in_console=True)
            # print(f'{type(data.data)} - {data.data}')

            self.world_timestamps = np.load(os.path.join(path,'world_timestamps.npy'))
            # print(f'{self.world_timestamps = }')
            # print(f'{len(self.world_timestamps) = }')

            print(f"[EyeDataHandler] Process all fiaxation data to match world timestamps")
            

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
                
                for world_index, item in enumerate(self.world_timestamps[last_world_index:]):
                    if item > current_timestamp:
                        frame = max(0,world_index-1)
                        self.fixation_start_world_frame[frame] = index
                        last_world_index = world_index
                        break
                
                # # print(f'[{worldf_index}] - {item}')

                # timestamp = dict_obj['timestamp']
                # index = np.where(world_timestamps == timestamp)
                # print(f'frame[{index}] - {timestamp}')

                # print("Topic:", dict_obj['topic'])
                # print("Normalized Position:", dict_obj['norm_pos'])
                # print("Dispersion:", dict_obj['dispersion'])
                # # print("Method:", dict_obj['method'])
                # # print("Base Data:", dict_obj['base_data'])
                # print("Timestamp:", dict_obj['timestamp'])
                # print("Duration:", dict_obj['duration'])
                # # print("Confidence:", dict_obj['confidence'])
                # # print("Gaze Point 3D:", dict_obj['gaze_point_3d'])
                # print("ID:", dict_obj['id'])

                # print('-----------------------')

            # print(f'{len(data.data) = }')
            # print(f'{len(data.timestamps) = }')
            # print(data.topics.__dir__())

            # print(f'{self.fixation_start_world_frame = }')

            with open(cache_file_path, 'wb') as f:
                pickle.dump({'fixation_start_world_frame': self.fixation_start_world_frame, 'data': self.data}, f)

        else:    
            print(f'[EyeDataHandler] Reload previous fixations data stored')
            with open(cache_file_path, 'rb') as f:
                data = pickle.load(f)
                self.fixation_start_world_frame = data['fixation_start_world_frame']
                self.data = data['data']

        execution_time = time.time() - start_time
        print(f"[EyeDataHandler] Finished process, took {execution_time:.2f} seconds")

    def step(self, frame_index):
        if frame_index in self.fixation_start_world_frame:
            timestamp_idx = self.fixation_start_world_frame[frame_index]
            # print(f"Data in frame {frame_index} is: {self.data[timestamp_idx]['norm_pos']}")
            return self.data[timestamp_idx]['norm_pos']
        # else:
            # print(f'No eye daat in frame {frame_index}')
        return None