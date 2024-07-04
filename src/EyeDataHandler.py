
import os
import numpy as np

from src.deps.file_methods import load_pldata_file

class EyeDataHandler:
    def __init__(self, path, topic_data='fixations'):
        data = load_pldata_file(directory=path, topic=topic_data, track_progress_in_console=True)
        # print(f'{type(data.data)} - {data.data}')

        world_timestamps = np.load(os.path.join(path,'world_timestamps.npy'))
        print(f'{world_timestamps = }')
        print(f'{len(world_timestamps) = }')


        for index, item in enumerate(data.data):
            dict_obj = dict(item)
            # # print(f'[{index}] - {item}')

            # timestamp = dict_obj['timestamp']
            # index = np.where(world_timestamps == timestamp)
            # print(f'frame[{index}] - {timestamp}')

            # print("Topic:", dict_obj['topic'])
            print("Normalized Position:", dict_obj['norm_pos'])
            print("Dispersion:", dict_obj['dispersion'])
            # print("Method:", dict_obj['method'])
            # print("Base Data:", dict_obj['base_data'])
            print("Timestamp:", dict_obj['timestamp'])
            print("Duration:", dict_obj['duration'])
            # print("Confidence:", dict_obj['confidence'])
            # print("Gaze Point 3D:", dict_obj['gaze_point_3d'])
            print("ID:", dict_obj['id'])

            print('-----------------------')

        print(f'{len(data.data) = }')
        print(f'{len(data.timestamps) = }')
        # print(data.topics.__dir__())