#!/usr/bin/env python3
# encoding: utf-8

import math
import random

import cv2 as cv
import numpy as np
from sklearn.decomposition import PCA

from src.BoardHandler import BoardHandler
from src.DistortionHandler import DistortionHandler
from src.utils import imshowMosaic

from src.detect_shapes import detectColorSquares, isSlot


# Hue in degrees (0-360); epsilon in degrees too
colors_dict = {'red':    {'h': 350,   'eps': 29}, 
              'green':  {'h': 125, 'eps': 35}, 
              'blue':   {'h': 220, 'eps': 35},
              'yellow': {'h': 50,  'eps': 28}}

colors_list = {'red': (0,0,255), 'green': (0,255,0), 'blue': (255,0,0), 'yellow': (0,255,255), 'black': (255,255,0)}

WINDOW_STREAM_CAMERA = 'Camera View'
WINDOW_STREAM_BOARD = 'Board View'
video_path = './data/world_cut.mp4'
fixations_data_path = './data/fixations_timestamps.npy'
board_configuration='./board_config.yaml'

patternSize = (7,4)
h_epsilon = 8


cv.namedWindow(WINDOW_STREAM_CAMERA, cv.WINDOW_AUTOSIZE)
cv.namedWindow(WINDOW_STREAM_BOARD, cv.WINDOW_AUTOSIZE)

def mouse_callback(event, x, y, flags, param):
    if event == cv.EVENT_LBUTTONDOWN:
        image = param
        hsv_image = cv.cvtColor(image, cv.COLOR_BGR2HSV_FULL)
        h, s, v = hsv_image[y, x]
        print(f'Hue: {int(h/255.0*360)}, Saturation: {s}, Value: {v}')

cv.setMouseCallback(WINDOW_STREAM_CAMERA, mouse_callback)
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



if __name__ == "__main__":


    stream = cv.VideoCapture(video_path)
    if not stream.isOpened():
        print(f"Could not open video {video_path}")
        exit()
        
    fps = stream.get(cv.CAP_PROP_FPS)
    frame_width = int(stream.get(cv.CAP_PROP_FRAME_WIDTH))
    frame_height = int(stream.get(cv.CAP_PROP_FRAME_HEIGHT))
    writer = cv.VideoWriter('./result.mp4', cv.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

    distortion_handler = DistortionHandler(calibration_json_path='camera_calib.json', 
                                           frame_width=frame_width, frame_height=frame_height)
    
    board_handler = BoardHandler(board_cfg_path=board_configuration, colors_dict=colors_dict, 
                                 colors_list=colors_list, distortion_handler=distortion_handler)
        
    board_metrics = {}
    while True:
        ret, original_image = stream.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        
        board_handler.step(original_image)

        ## FIXATION COORDINATES FOR THIS FRAME?
        # desnormalized_x = int(0.5 * capture.shape[0])
        # desnormalized_y = int(0.5 * capture.shape[1])
        # corrected_coord = correctCoordinates((desnormalized_x, desnormalized_y))[0][0]
        corrected_coord = np.array([(int(random.random()*board_handler.board_view.shape[0]), int(random.random()*board_handler.board_view.shape[1]))])
        
        color, shape, slot = board_handler.getPixelInfo(corrected_coord)

        board_view_cfg, board_view_detected = board_handler.getVisualization()
        image_board_cfg, image_board_detected = board_handler.getUndistortedVisualization(original_image)

        # Update board metrics
        if color is not None:
            if color not in board_metrics:
                board_metrics[color] = {shape: {True: 0, False: 0}}
            if shape not in board_metrics[color]:
                board_metrics[color][shape] = {True: 0, False: 0}

            board_metrics[color][shape][slot] += 1

        def buildMosaic(titles_list, images_list, rows, cols, window_name, resize = 1):
            for index, resized_image in enumerate(images_list):
                images_list[index] = cv.resize(resized_image, (int(frame_width*resize), int(frame_height*resize)))

            imshowMosaic(titles_list=titles_list, 
                        images_list=images_list, 
                        rows=rows, cols=cols, window_name=window_name)
        
        buildMosaic(titles_list=['Board Cfg', 'Board Detected'], 
                     images_list=[board_view_cfg, board_view_detected], 
                     rows=2, cols=1, window_name=WINDOW_STREAM_BOARD, resize = 1/2)
        
        buildMosaic(titles_list=['Complete Cfg', 'Complete Detected'], 
                     images_list=[image_board_cfg, image_board_detected], 
                     rows=2, cols=1, window_name=WINDOW_STREAM_CAMERA, resize = 1/2)


        # resized_frame = cv.resize(display_image, (frame_width, frame_height))
        # writer.write(resized_frame)

        # check keystroke to exit (image window must be on focus)
        key = cv.pollKey()
        # key = cv.waitKey(0)
        if key == ord('q') or key == ord('Q') or key == 27:
            break

    if stream.isOpened():  stream.release()
    if writer.isOpened():  writer.release()
    cv.destroyAllWindows()


    for color, shapes_dict in board_metrics.items():
        time_color = 0
        for shape, slot_dict in shapes_dict.items():
            for slot, num in slot_dict.items():
                num = float(num)/float(fps)
                time_color += num

        print(f'------------------------------')
        print(f'Color: {color}: {time_color}s')
        for shape, slot_dict in shapes_dict.items():
            for slot, num in slot_dict.items():
                print(f'Shape: {shape} ({slot}): {num}s')
