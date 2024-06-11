#!/usr/bin/env python3
# encoding: utf-8

import math
import random

import cv2 as cv
import numpy as np
from sklearn.decomposition import PCA

from src.image_correction import initCalibrationData, correctImage, correctCoordinates
from src.load_board_config import getBoardData, projectBoardMatrix, projectBoardConfig, getCellIndex
from src.detect_shapes import detectBoardContour, detectColorSquares, isSlot

# Hue in degrees (0-360); epsilon in degrees too
color_dict = {'red':    {'h': 350,   'eps': 29}, 
              'green':  {'h': 125, 'eps': 35}, 
              'blue':   {'h': 220, 'eps': 35},
              'yellow': {'h': 50,  'eps': 28}}

colors_list = {'red': (0,0,255), 'green': (0,255,0), 'blue': (255,0,0), 'yellow': (0,255,255), 'black': (255,255,0)}

WINDOW_STREAM = '(W1) Video'
video_path = './data/world_cut.mp4'
fixations_data_path = './data/fixations_timestamps.npy'
board_configuration='./board_config.yaml'

patternSize = (7,4)
h_epsilon = 8


cv.namedWindow(WINDOW_STREAM, cv.WINDOW_AUTOSIZE)

def mouse_callback(event, x, y, flags, param):
    if event == cv.EVENT_LBUTTONDOWN:
        # Convertir la imagen a HSV
        hsv_image = cv.cvtColor(capture, cv.COLOR_BGR2HSV_FULL)
        
        # Obtener los valores HSV del píxel
        h, s, v = hsv_image[y, x]
        
        print(f'Hue: {int(h/255.0*360)}, Saturation: {s}, Value: {v}')

cv.setMouseCallback(WINDOW_STREAM, mouse_callback)


# cv.namedWindow('yellow_mask', cv.WINDOW_AUTOSIZE)
# cv.namedWindow('red_mask', cv.WINDOW_AUTOSIZE)
# cv.namedWindow('green_mask', cv.WINDOW_AUTOSIZE)
# cv.namedWindow('blue_mask', cv.WINDOW_AUTOSIZE)
# cv.setMouseCallback('yellow_mask', mouse_callback)
# cv.setMouseCallback('red_mask', mouse_callback)
# cv.setMouseCallback('green_mask', mouse_callback)
# cv.setMouseCallback('blue_mask', mouse_callback)

def ComputePixelsPerMilimeters(approx_contour, real_width_mm, real_height_mm):
    width_px = np.linalg.norm(approx_contour[0][0] - approx_contour[1][0])
    height_px = np.linalg.norm(approx_contour[1][0] - approx_contour[2][0])
    
    pixels_per_mm_width = width_px / real_width_mm
    pixels_per_mm_height = height_px / real_height_mm

    pixels_per_mm = (pixels_per_mm_width + pixels_per_mm_height) / 2
    return pixels_per_mm



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

    initCalibrationData(frame_height, frame_width, calibration_json = 'camera_calib.json')

    board_size, board_size_mm, board_data_dict = getBoardData(board_configuration)
        
    fixations = {}
    while True:
        ret, capture = stream.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        
        # Distortion and perspective correction
        capture, display_image = correctImage(capture=capture)

        board_contour = detectBoardContour(capture, display_image)
        if board_contour is not None and len(board_contour) != 0:
            cell_matrix, cell_width, cell_height = projectBoardMatrix(board_contour, board_size, display_image = None)
            board_data_dict = projectBoardConfig(cell_matrix, cell_width, cell_height, board_data_dict, display_image = None, colors_list=colors_list)
            

            # desnormalized_x = int(0.5 * capture.shape[0])
            # desnormalized_y = int(0.5 * capture.shape[1])
            # corrected_coord = correctCoordinates((desnormalized_x, desnormalized_y))[0][0]
            corrected_coord = (int(random.random()*display_image.shape[0]), int(random.random()*display_image.shape[1]))
            idx = getCellIndex(corrected_coord, cell_matrix, cell_width, cell_height)
            if idx[0] is not None:
                print(f"Fixation detected in: {board_data_dict[idx]}")
                color = board_data_dict[idx][0]
                shape = board_data_dict[idx][1]
                slot = board_data_dict[idx][2]

                if color not in fixations:
                    fixations[color] = {shape: {True: 0, False: 0}}
                if shape not in fixations[color]:
                    fixations[color][shape] = {True: 0, False: 0}

                fixations[color][shape][slot] += 1
                cv.circle(display_image, (int(corrected_coord[0]),int(corrected_coord[1])), radius=10, color=(0,0,255), thickness=-1)
        

        ## Detection of squares and slots :)
        # might be unnecesary
        detected_board_data = []
        contours_dict = detectColorSquares(capture, color_dict=color_dict, colors_list=colors_list, display_image=None)
        for color, contour_list in contours_dict.items():
            for contour in contour_list:
                # print(f'Detect shape for {color} and contour {contour}')
                is_slot, center = isSlot(capture, contour, color_dict[color], colors_list[color], display_image=None)
                detected_board_data.append([is_slot, center])

        cv.imshow(WINDOW_STREAM, display_image)

        resized_frame = cv.resize(display_image, (frame_width, frame_height))
        writer.write(resized_frame)

        # check keystroke to exit (image window must be on focus)
        key = cv.pollKey()
        # key = cv.waitKey(0)
        if key == ord('q') or key == ord('Q') or key == 27:
            break

    if stream.isOpened():  stream.release()
    if writer.isOpened():  writer.release()
    cv.destroyAllWindows()


    for color, shapes_dict in fixations.items():
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
