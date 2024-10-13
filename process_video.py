#!/usr/bin/env python3
# encoding: utf-8

import os
import math
import random
import argparse

import cv2 as cv
import numpy as np
from sklearn.decomposition import PCA

from src.detect_shapes import detectColorSquares, isSlot

from src.BoardHandler import BoardHandler
from src.DistortionHandler import DistortionHandler
from src.PanelHandler import PanelHandler
# from src.EyeDataHandler import EyeDataHandlerCSV as EyeDataHandler
from src.EyeDataHandler import EyeDataHandlerPLDATA as EyeDataHandler
from src.ArucoBoardHandler import ARUCOColorCorrection
from src.StateMachineHandler import StateMachine


# Hue in degrees (0-360); epsilon in degrees too
colors_dict = {'red':    {'h': 350, 'eps': 29}, 
               'green':  {'h': 125, 'eps': 35}, 
               'blue':   {'h': 220, 'eps': 35},
               'yellow': {'h': 50,  'eps': 28}}

colors_list = {'red': (0,0,255), 'green': (0,255,0), 'blue': (255,0,0), 'yellow': (0,255,255), 'board': (255,255,0)}

participant_id = '00001'

parser = argparse.ArgumentParser(description='Process PupilLabs data with board')
parser.add_argument('-p', dest='participant', type=str, default=participant_id, metavar='id', help='Participant folder id')
participant_id = parser.parse_args().participant

WINDOW_STREAM_CAMERA = f'Camera View {participant_id}'
WINDOW_STREAM_BOARD = f'Board View {participant_id}'
participant_path = f'/home/quique/eeha/eyes_board_color/data/{participant_id}/'
data_path = participant_path #os.path.join(participant_path,'exports', '000')
video_path = os.path.join(participant_path,'world.mp4')
output_path = f'/home/quique/eeha/eyes_board_color/output/{participant_id}/'

CURRENT_FILE_PATH = os.path.dirname(os.path.abspath(__file__))
print(f"Script root folder: {CURRENT_FILE_PATH}")
game_configuration= os.path.join(CURRENT_FILE_PATH,'game_config.yaml')
game_aruco_board_cfg= os.path.join(CURRENT_FILE_PATH,'game_aruco_board.yaml')
samples_configuration= os.path.join(CURRENT_FILE_PATH,'sample_shape_cfg')
eye_data_topic = 'gaze' #'fixations'
frame_speed_multiplier = 1 # process one frame each N to go faster

init_capture_idx = 0 #4300 #18700

enable_visualization = True


# cv.namedWindow(WINDOW_STREAM_CAMERA, cv.WINDOW_AUTOSIZE)
cv.namedWindow(WINDOW_STREAM_BOARD, cv.WINDOW_AUTOSIZE)

mouse_callback_image = None
def mouse_callback(event, x, y, flags, param):
    global mouse_callback_image
    if event == cv.EVENT_LBUTTONDOWN and mouse_callback_image is not None:
        image = mouse_callback_image
        hsv_image = cv.cvtColor(image, cv.COLOR_BGR2HSV_FULL)
        h, s, v = hsv_image[y, x]
        print(f'Hue: {int(h/255.0*360)}, Saturation: {s}, Value: {v}')
    if event == cv.EVENT_LBUTTONDOWN and mouse_callback_image is None:
        print(f'No image provided to detect HSV components.')

# cv.setMouseCallback(WINDOW_STREAM_CAMERA, mouse_callback)
cv.setMouseCallback(WINDOW_STREAM_BOARD, mouse_callback)




# Contours detected in image and data dict of the board
def checkBoardMatch(contours, data_dict, board_size):

    coordinates = {}
    center_list = []
    coordinates_idx = {}
    
    for color, contours in contours.items():
        for contour in contours:
            center = projectCenter(contour)
            if center is None:
                continue
            coordinates[tuple(center)] = {'color': color, 'contour':contour}
            center_list.append(center)

    # Find real dimensions to de-rotate board (PCA over center_list)
    center_list = np.array(center_list)
    pca = PCA(n_components=2)
    pca.fit(center_list)

    aligned_points = pca.transform(center_list)
    reconstructed_points = pca.inverse_transform(aligned_points)    
    reconstructed_points = np.round(reconstructed_points).astype(int)
   
    max_vals = np.max(reconstructed_points, axis=0)
    min_vals = np.min(reconstructed_points, axis=0)

    print(f"{len(reconstructed_points)}")
    print(f"{min_vals = }; {max_vals = }")

    # Get averaged min distances to compute distance in X and Y between boxes 
    dist_x = []
    dist_y = []
    min_x, min_y = math.inf, math.inf
    for i, point1 in enumerate(reconstructed_points):
        for j, point2 in enumerate(reconstructed_points):
            if i == j: # Skip same point
                continue
            distance_x = abs(point1[0] - point2[0])
            distance_y = abs(point1[1] - point2[1])

            min_x = min(distance_x, min_x)
            min_y = min(distance_y, min_y)

        dist_x.append(min_x)
        dist_y.append(min_y)
        min_x, min_y = math.inf, math.inf

    dist = [np.average(dist_x),np.average(dist_y)]

    matriz = np.full(board_size, None)
    for i, point in enumerate(reconstructed_points):
        index = int(point[0] - min_vals[0]- dist[0]), int(point[1] - min_vals[1]- dist[1])
        print(f"{point = }")
        print(f"[{point[0] - min_vals[0]}, {point[1] - min_vals[1]}]: {center_list[i]}")
        matriz[index] = tuple(center_list[i])


    print(matriz)
    exit()

    for center, data in coordinates.items():
        index = tuple((0,0))
        coordinates_idx[index] = data.update({'center':center})
        print(f"{index}: 'center':{data['center']}, 'color': {data['color']}")

        if index not in data_dict:
            print(f"ERROR could not locate {index} in board dict")
            continue

        if not data['color'] is data_dict[index][0]: # Check if color matches
            print(f"Color does not match: {data['color'] = }; {data_dict[index][0] = }")




def processVideo(video_path):
    global init_capture_idx, mouse_callback_image

    stream = cv.VideoCapture(video_path)
    if not stream.isOpened():
        print(f"Could not open video {video_path}")
        exit()

    total_frames = int(stream.get(cv.CAP_PROP_FRAME_COUNT))
    print(f'[processVideo::{participant_id}] Processing participant: {participant_id}')
    print(f'[processVideo::{participant_id}] Processing: {video_path}')
    print(f'[processVideo::{participant_id}] Video with: {total_frames} frames')

    fps = stream.get(cv.CAP_PROP_FPS)
    frame_width = int(stream.get(cv.CAP_PROP_FRAME_WIDTH))
    frame_height = int(stream.get(cv.CAP_PROP_FRAME_HEIGHT))
    print(f"[processVideo::{participant_id}] Processing video of {fps} FPS and {frame_width = }; {frame_height = }")
    print(f"[processVideo::{participant_id}] Processing video duration {total_frames/fps} seconds")
    writer = cv.VideoWriter('./result.mp4', cv.VideoWriter_fourcc(*'mp4v'), int(fps/frame_speed_multiplier), (frame_width*3, frame_height*2))

    distortion_handler = DistortionHandler(calibration_json_path='camera_calib.json', 
                                           frame_width=frame_width, frame_height=frame_height)
    
    board_handler = BoardHandler(aruco_board_cfg_path = game_aruco_board_cfg,
                                 game_cfg_path=game_configuration, colors_dict=colors_dict, 
                                 colors_list=colors_list, distortion_handler=distortion_handler)
    
    panel_handler = PanelHandler(panel_configuration_path=samples_configuration, colors_dict=colors_dict,
                                 colors_list=colors_list, distortion_handler=distortion_handler)
    
    eye_data_handler = EyeDataHandler(root_path=participant_path, data_path=data_path, video_fps=fps, topic_data=eye_data_topic)

    state_machine_handler = StateMachine(board_handler,panel_handler,eye_data_handler, video_fps=fps)
    
    capture_idx = init_capture_idx
    stream.set(cv.CAP_PROP_POS_FRAMES, capture_idx)

    while True:
        if capture_idx >= total_frames:
            print(f"[processVideo::{participant_id}] End of video detected :)")
            break

        stream.set(cv.CAP_PROP_POS_FRAMES, capture_idx)
        ret, original_image = stream.read()
        original_image = ARUCOColorCorrection(original_image)
        if not ret:
            print(f"[processVideo::{participant_id}] Can't receive frame (stream end?). Exiting ...")
            break
        

        state_machine_handler.step(original_image, capture_idx)

        if enable_visualization:
            mosaic, log_frame = state_machine_handler.visualization(original_image=original_image, capture_idx=capture_idx, 
                                                                    last_capture_idx=total_frames, frame_width=frame_width, 
                                                                    frame_height=frame_height, participan_id=participant_id)
            mouse_callback_image = mosaic
            cv.imshow(WINDOW_STREAM_BOARD, mosaic)
            writer.write(log_frame)
        else:
            print(f"FPS: {state_machine_handler.tm.getFPS():.1f}")

        # check keystroke to exit (image window must be on focus)
        key = cv.pollKey()

        # if norm_coord is not None:
        #     key = cv.waitKey(0)

        if key == ord('q') or key == ord('Q') or key == 27:
            break

        capture_idx+=frame_speed_multiplier*state_machine_handler.getFrameMultiplier()

    if stream.isOpened():  stream.release()
    if writer.isOpened():  writer.release()
    cv.destroyAllWindows()


    os.makedirs(output_path, exist_ok=True)
    state_machine_handler.log_results(output_path=output_path)
    
    


if __name__ == "__main__":
    processVideo(video_path)