#!/usr/bin/env python3
# encoding: utf-8

import os
import math
import random

import cv2 as cv
import numpy as np
from sklearn.decomposition import PCA

from src.detect_shapes import detectColorSquares, isSlot

from src.BoardHandler import BoardHandler
from src.DistortionHandler import DistortionHandler
from src.PanelHandler import PanelHandler
from src.EyeDataHandler import EyeDataHandler
from src.ArucoBoardHandler import ARUCOColorCorrection
from src.StateMachineHandler import StateMachine


# Hue in degrees (0-360); epsilon in degrees too
colors_dict = {'red':    {'h': 350, 'eps': 29}, 
               'green':  {'h': 125, 'eps': 35}, 
               'blue':   {'h': 220, 'eps': 35},
               'yellow': {'h': 50,  'eps': 28}}

colors_list = {'red': (0,0,255), 'green': (0,255,0), 'blue': (255,0,0), 'yellow': (0,255,255), 'board': (255,255,0)}

WINDOW_STREAM_CAMERA = 'Camera View'
WINDOW_STREAM_BOARD = 'Board View'
# video_path = './data/world_cut.mp4'
# data_path = '/home/quique/eeha/eyes_board_color/data/011-20240624T152508Z-001/011/'
data_path = '/home/quique/eeha/eyes_board_color/data/0001/'
video_path = os.path.join(data_path,'world.mp4')
game_configuration='./game_config.yaml'
game_aruco_board_cfg='./game_aruco_board.yaml'
samples_configuration='./sample_shape_cfg'
eye_data_topic = 'fixations'
frame_speed_multiplier = 3 # process one frame each N to go faster

patternSize = (7,4)
h_epsilon = 8
init_capture_idx = 5600


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

    fps = stream.get(cv.CAP_PROP_FPS)
    frame_width = int(stream.get(cv.CAP_PROP_FRAME_WIDTH))
    frame_height = int(stream.get(cv.CAP_PROP_FRAME_HEIGHT))
    print(f"Processing video of {fps} FPS and {frame_width = }; {frame_height = }")
    writer = cv.VideoWriter('./result.mp4', cv.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

    distortion_handler = DistortionHandler(calibration_json_path='camera_calib.json', 
                                           frame_width=frame_width, frame_height=frame_height)
    
    board_handler = BoardHandler(aruco_board_cfg_path = game_aruco_board_cfg,
                                 game_cfg_path=game_configuration, colors_dict=colors_dict, 
                                 colors_list=colors_list, distortion_handler=distortion_handler)
    
    panel_handler = PanelHandler(panel_configuration_path=samples_configuration, colors_dict=colors_dict,
                                 colors_list=colors_list, distortion_handler=distortion_handler)
    
    eye_data_handler = EyeDataHandler(data_path, eye_data_topic)

    state_machine_handler = StateMachine(board_handler,panel_handler,eye_data_handler)
    
    capture_idx = init_capture_idx
    stream.set(cv.CAP_PROP_POS_FRAMES, capture_idx)

    while True:
        stream.set(cv.CAP_PROP_FRAME_COUNT, capture_idx)
        ret, original_image = stream.read()
        original_image = ARUCOColorCorrection(original_image)
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        

        state_machine_handler.step(original_image, capture_idx)

        mosaic, log_frame = state_machine_handler.visualization(original_image, capture_idx, frame_width, frame_height)
        mouse_callback_image = mosaic
        cv.imshow(WINDOW_STREAM_BOARD, mosaic)
        writer.write(log_frame)


        # check keystroke to exit (image window must be on focus)
        key = cv.pollKey()

        # if norm_coord is not None:
        #     key = cv.waitKey(0)

        if key == ord('q') or key == ord('Q') or key == 27:
            break

        capture_idx+=frame_speed_multiplier*state_machine_handler.getFrameMultiplier()

    state_machine_handler.print_results(fps)

    if stream.isOpened():  stream.release()
    if writer.isOpened():  writer.release()
    cv.destroyAllWindows()



if __name__ == "__main__":
    stream = cv.VideoCapture(video_path)
    frames = stream.get(cv.CAP_PROP_FRAME_COUNT)
    print(f'{frames}')

    processVideo(video_path)