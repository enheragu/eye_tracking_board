#!/usr/bin/env python3
# encoding: utf-8

"""
    Script to project data into the original video. Just fixation or gaze data
    from pldata/CSV
"""

import os

import cv2 as cv
import numpy as np
from tqdm import tqdm

# from src.EyeDataHandler import EyeDataHandlerCSV as EyeDataHandler
from src.EyeDataHandler import EyeDataHandlerPLDATA as EyeDataHandler
import process_video
# from src.ThreadVideoStream import ThreadVideoCapture, ThreadVideoWriter
from src.utils import log

data_path = process_video.participant_path
# data_path = os.path.join(process_video.participant_path,'exports', '000')
eye_data_topic = 'gaze'

output_video_path = os.path.join(process_video.output_path,f'data_projection_{process_video.participant_id}.mp4')

if __name__ == "__main__":
    # No windows needed here...

    stream = cv.VideoCapture(process_video.video_path)
    # stream = ThreadVideoCapture(process_video.video_path)
    if not stream.isOpened():
        log(f"Could not open video {process_video.video_path}")
        exit()

    total_frames = int(stream.get(cv.CAP_PROP_FRAME_COUNT))
    fps = stream.get(cv.CAP_PROP_FPS)
    frame_width = int(stream.get(cv.CAP_PROP_FRAME_WIDTH))
    frame_height = int(stream.get(cv.CAP_PROP_FRAME_HEIGHT))
    writer = cv.VideoWriter(output_video_path, cv.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))
    # writer = ThreadVideoWriter(f'./projection_{process_video.participant_id}.mp4', cv.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))
    eye_data_handler = EyeDataHandler(root_path=process_video.participant_path, data_path=data_path, video_fps=fps, topic_data=eye_data_topic)


    # stream.start()
    # writer.start()
    with tqdm(total=total_frames, desc=f"Frames from {process_video.participant_id}") as pbar:
        for capture_idx in range(total_frames):
            ret, original_image = stream.read()
            if not ret:
                log(f"[processVideo::{process_video.participant_id}] Can't receive frame (stream end?). Exiting ...")
                break

            norm_coord_list = eye_data_handler.step(capture_idx)
            desnormalized_coord_list = []
            for norm_coord in norm_coord_list:
                # log(f'{capture_idx =}')
                desnormalized_x = int(norm_coord[0] * original_image.shape[1])
                desnormalized_y = int(norm_coord[1] * original_image.shape[0])
                desnormalized_coord_list.append(np.array([[desnormalized_x, desnormalized_y]]))

            if norm_coord_list:
                for desnormalized_coord in desnormalized_coord_list:
                    cv.circle(original_image, desnormalized_coord[0], radius=10, color=(255,255,0), thickness=-1)
                    cv.circle(original_image, desnormalized_coord[0], radius=5, color=(0,0,255), thickness=-1)


            text = f'Frame: {capture_idx}/{total_frames}'
            font = cv.FONT_HERSHEY_SIMPLEX
            scale = 0.4
            thickness = 1
            text_size, _ = cv.getTextSize(text, font, scale, thickness)
            text_width, text_height = text_size
            x = 1#original_image.shape[1] - text_width - 1  # 10 pixel margin
            y = text_height + 1  # 10 pixel margin
            cv.putText(original_image, text, (x, y), font, scale, color=(0,0,0), thickness=thickness)

            writer.write(original_image)            
            pbar.update(1) 

            cv.imshow('Projection',original_image)
            key = cv.waitKey()
            if key == ord('q') or key == ord('Q') or key == 27:
                exit()

    if stream.isOpened():  stream.release()
    if writer.isOpened():  writer.release()

    log(f"Video stored to: {output_video_path}")