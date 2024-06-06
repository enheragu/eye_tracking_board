#!/usr/bin/env python3
# encoding: utf-8

import os
import math
import json

import cv2 as cv
import numpy as np
from sklearn.decomposition import PCA

import yaml
from yaml.loader import SafeLoader

from detect_aruco import drawArucos

# Hue in degrees (0-360); epsilon in degrees too
color_dict = {'red':    {'h': 350,   'eps': 29}, 
              'green':  {'h': 125, 'eps': 35}, 
              'blue':   {'h': 220, 'eps': 35},
              'yellow': {'h': 50,  'eps': 30}}

colors_list = {'red': (0,0,255), 'green': (0,255,0), 'blue': (255,0,0), 'yellow': (0,255,255), 'black': (0,0,0)}

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


def projectCenter(contour):
    center = None
    M = cv.moments(contour)
    if M["m00"] != 0:
        # Calcular las coordenadas del centroide
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        center_x = int((M['m10'] / M['m00']))
        center_y = int((M['m01'] / M['m00']))
        center = (center_x, center_y)
    return center

def centerHueChannel(hue_channel, new_center):
    lookUpTable = np.zeros(256, dtype=np.uint8)
    
    for i in range(len(lookUpTable)):
        lookUpTable[i] = (i+128-new_center) % 256

    return cv.LUT(hue_channel, lookUpTable)

def getMaskHue(hue, sat, intensity, h_ref, h_epsilon, s_margins = [5,255], v_margins = [60,215]):
    h_ref = int(h_ref/360.0*255.0)
    h_epsilon = int(h_epsilon/360.0*255.0)

    hue_new = centerHueChannel(hue, h_ref)
    hsv_new = cv.merge((hue_new, sat, intensity))

    res = cv.inRange(hsv_new, 
                        tuple([128-h_epsilon,s_margins[0],v_margins[0]]), 
                        tuple([128+h_epsilon,s_margins[1],v_margins[1]]))
    kernel = cv.getStructuringElement(cv.MORPH_CROSS, (3, 3))   # [[0,1,0], [1,1,1], [0,1,0]]
    res = cv.morphologyEx(res, cv.MORPH_OPEN, kernel,  iterations=2)
    res = cv.morphologyEx(res, cv.MORPH_CLOSE, kernel, iterations=2)
    
    return res

def checkShape(contour, area_filter = [900,10000]):
    
    shapes = {'triangle': {'sides': 3, 'aspect_ratio': None, 'circularity': None},
              'square': {'sides': 4, 'aspect_ratio': [0.8,1.1], 'circularity': None},
              'rectangle': {'sides': 4, 'aspect_ratio': [0, math.inf], 'circularity': None}, # if not square
              'hexagon': {'sides': 6, 'aspect_ratio': None, 'circularity': [0.4, 0.8]},
              'trapezoid': {'sides': 4, 'aspect_ratio': None, 'circularity': [0, 0.6]},
              'circle': {'sides': None, 'aspect_ratio': None, 'circularity': [0.85, 1]}
            }
    
    perimeter = cv.arcLength(contour, True)
    if not perimeter > 0.1:
        return None, None

    approximate = cv.approxPolyDP(contour, .04 * perimeter, True)
    area = cv.contourArea(contour)

    if area < area_filter[0] or area > area_filter[1]:
        return None, None

    x, y, w, h = cv.boundingRect(contour)
    aspect_ratio = float(w) / h

    circularity = 4 * np.pi * area / (perimeter * perimeter)
    for shape, data in shapes.items():
        if data['sides'] is not None and len(approximate) != data['sides']:
            continue

        if data['aspect_ratio'] is not None and not (data['aspect_ratio'][0] < aspect_ratio < data['aspect_ratio'][1]):
            continue
        
        if data['circularity'] is not None and not data['circularity'][0] < circularity < data['circularity'][1]:
            continue

        return shape, approximate
    
    return None, None

def detectBoardTransform(image, display_image):
    global margin_color


    # gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    # # gray = claheEqualization(gray)
    # edge_image = cv.Canny(gray, threshold1=50, threshold2=200)

    # # cv.imshow(f'border_mask', edge_image)

    # return []

    hue, sat, intensity = cv.split(cv.cvtColor(image, cv.COLOR_BGR2HSV_FULL))
    # Take any hue with low brightness
    res = getMaskHue(hue, sat, intensity, h_ref=0, h_epsilon=180, s_margins=[0,255], v_margins = [0,70])
    # cv.imshow(f'border_mask', res)

    edge_image = cv.Canny(res, threshold1=50, threshold2=200)
    contours, hierarchy = cv.findContours(edge_image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    
    cv.imshow(f'border_edges', edge_image)

    borders = []
    for contour in contours:
        shape, approx = checkShape(contour, area_filter=[1000, math.inf])
        if shape == 'rectangle':
            borders.append(approx)
    display_image = cv.drawContours(display_image, borders, -1, colors_list['black'], 5)
    return borders

def detectColorSquares(image, display_image):
    global color_dict, colors_list, color_dict_epsilon
    
    contours_filtered = dict()

    hue, sat, intensity = cv.split(cv.cvtColor(image, cv.COLOR_BGR2HSV_FULL))
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    for color, h_ref in color_dict.items():
        res = getMaskHue(hue, sat, intensity, h_ref['h'], h_ref['eps'])
        edge_image = cv.Canny(res, threshold1=50, threshold2=200)
        
        # masked_image = cv.bitwise_not(res)
        # masked_image = cv.bitwise_and(image, image, mask=masked_image)
        # cv.imshow(f'{color}_mask',masked_image)
        # cv.imshow(f'{color}_mask', edge_image)
        

        # lines = cv.HoughLinesP(edge_image, 1, np.pi/180, threshold=100, minLineLength=5, maxLineGap=10)
        
        # if lines is None:
        #     continue

        # for line in lines:
        #     x1,y1,x2,y2 = line[0]
        #     cv.line(image, (x1,y1), (x2,y2), colors_list[color], 2)


        contours, hierarchy = cv.findContours(edge_image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        squares = list()
        for contour in contours:
            shape, approx = checkShape(contour, area_filter=[1000, math.inf])
            if shape == 'square':
                squares.append(approx)
        
        display_image = cv.drawContours(display_image, squares, -1, colors_list[color], 2)
        contours_filtered[color] = squares
    
    return contours_filtered


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


def claheEqualization(channel):
    clahe = cv.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    return clahe.apply(channel)

if __name__ == "__main__":

    ## Get camera calibration data
    cameraMatrix = None
    distCoeffs = None
    if os.path.exists('camera_calib.json'):
        try:
            with open('camera_calib.json') as file:
                data = json.load(file)

            cameraMatrix = np.array(data['camera_matrix'])
            distCoeffs = np.array(data['distortion_coefficients'])

            print('Camera matrix:', cameraMatrix)
            print('distortion_coefficients:', distCoeffs)

        except:
            print("Calibration file not valid")



    stream = cv.VideoCapture(video_path)
    if not stream.isOpened():
        print(f"Could not open video {video_path}")
        exit()
        
    fps = stream.get(cv.CAP_PROP_FPS)
    frame_width = int(stream.get(cv.CAP_PROP_FRAME_WIDTH))
    frame_height = int(stream.get(cv.CAP_PROP_FRAME_HEIGHT))
    writer = cv.VideoWriter('./result.mp4', cv.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))
    

    # If calibration data is available, undistort the image
    if cameraMatrix is not None:
        h, w = frame_height, frame_width
        newcameramtx, roiundistort = cv.getOptimalNewCameraMatrix(cameraMatrix, distCoeffs, (w,h), alpha=0, newImgSize=(w,h))
        mapx, mapy = cv.initUndistortRectifyMap(cameraMatrix, distCoeffs, None, newcameramtx, (w,h), 5)


    board_data_dict = {}
    with open(board_configuration) as file:
        data = yaml.load(file, Loader=SafeLoader)
        board_data_dict = {tuple(map(int, key.split(','))): value for key, value in data['board_config'].items()}
        board_size = data['board_size']
        

    while True:
        ret, capture = stream.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        
        # If calibration data is available, undistort the image
        if cameraMatrix is not None:
            dst = cv.remap(capture, mapx, mapy, cv.INTER_LINEAR)
 
            # crop the image to given roi (alpha = 0 needs no roi?¿)
            x, y, w, h = roiundistort
            capture = capture[y:y+h, x:x+w]

        # x0,y0 = 210,210
        # x1,y1 = 990,690
        # capture = capture[y0:y1, x0:x1]

        display_image = capture.copy()
        contours = detectColorSquares(capture, display_image)
        borders = detectBoardTransform(capture, display_image)

        # checkBoardMatch(contours, board_data_dict, board_size)

        cv.imshow(WINDOW_STREAM, display_image)
        writer.write(display_image)

        # check keystroke to exit (image window must be on focus)
        key = cv.pollKey()
        key = cv.waitKey(0)
        if key == ord('q') or key == ord('Q') or key == 27:
            break


    cv.destroyAllWindows()
    if stream.isOpened():  stream.release()
    if writer.isOpened():  writer.release()
