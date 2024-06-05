#!/usr/bin/env python3
# encoding: utf-8

import cv2 as cv
import numpy as np

from detect_aruco import drawArucos


color_dict = {'red':    {'h':3, 'eps':10}, 
              'green':  {'h':82, 'eps':15}, 
              'blue':   {'h':160, 'eps':20},
              'yellow': {'h': 37, 'eps':8}}

colors_list = {'red': (0,0,255), 'green': (0,255,0), 'blue': (255,0,0), 'yellow': (0,255,255)}

WINDOW_STREAM = '(W1) Video'
video_path = './data/world.mp4'
fixations_data_path = './data/fixations_timestamps.npy'

patternSize = (7,4)
h_epsilon = 8

stream = cv.VideoCapture(video_path)
if not stream.isOpened():
    print(f"Could not open video {video_path}")
    exit()

cv.namedWindow(WINDOW_STREAM, cv.WINDOW_AUTOSIZE)

def mouse_callback(event, x, y, flags, param):
    if event == cv.EVENT_LBUTTONDOWN:
        # Convertir la imagen a HSV
        hsv_image = cv.cvtColor(capture, cv.COLOR_BGR2HSV_FULL)
        
        # Obtener los valores HSV del píxel
        h, s, v = hsv_image[y, x]
        
        print(f'Hue: {h}, Saturation: {s}, Value: {v}')

cv.setMouseCallback(WINDOW_STREAM, mouse_callback)


def detect_color_contour(image):
    global color_dict, colors_list, color_dict_epsilon
    
    contours_filtered = dict()

    hue, sat, intensity = cv.split(cv.cvtColor(image, cv.COLOR_BGR2HSV_FULL))
    for color, h_ref in color_dict.items():
        lookUpTable = np.zeros(256, dtype=np.uint8)
        
        for i in range(len(lookUpTable)):
            lookUpTable[i] = (i+128-h_ref['h']) % 256

        hue_new = cv.LUT(hue, lookUpTable)
        hsv_new = cv.merge((hue_new, sat, intensity))

        res = cv.inRange(hsv_new, 
                         tuple([128-h_ref['eps'],0,0]), 
                         tuple([128+h_ref['eps'],255,255]))
        kernel = cv.getStructuringElement(cv.MORPH_CROSS, (3, 3))   # [[0,1,0], [1,1,1], [0,1,0]]
        res = cv.morphologyEx(res, cv.MORPH_OPEN, kernel,  iterations=2)
        res = cv.morphologyEx(res, cv.MORPH_CLOSE, kernel, iterations=2)

        edge_image = cv.Canny(res, threshold1=50, threshold2=200)
        contours, hierarchy = cv.findContours(edge_image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

        cv.imshow(f'{color}_mask', res)
        
        squares = list()
        for contour in contours:
            perimeter = cv.arcLength(contour, True)
            if not perimeter > 0.1:
                continue

            approximate = cv.approxPolyDP(contour, .04 * perimeter, True)
            if len(approximate) != 4:
                continue

            area = cv.contourArea(contour)
            if area < 900 or area > 10000:
                continue

            x, y, w, h = cv.boundingRect(contour)
            aspect_ratio = float(w) / h
            if aspect_ratio < 0.8 or aspect_ratio > 1.2:
                continue
            
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            if circularity < 0.6:
                continue

            squares.append(contour)
        
        image = cv.drawContours(image, squares, -1, colors_list[color], 2)
        contours_filtered[color] = squares
    
    return contours_filtered


def detectShapes(image, contours):
    pass

if __name__ == "__main__":

    print(np.load(fixations_data_path))
    
    fps = stream.get(cv.CAP_PROP_FPS)
    frame_width = int(stream.get(cv.CAP_PROP_FRAME_WIDTH))
    frame_height = int(stream.get(cv.CAP_PROP_FRAME_HEIGHT))
    writer = cv.VideoWriter('./result.mp4', cv.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))
    
    while True:
        ret, capture = stream.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        # x0,y0 = 210,210
        # x1,y1 = 990,690
        # capture = capture[y0:y1, x0:x1]
        contours = detect_color_contour(capture)

        cv.imshow(WINDOW_STREAM, capture)
        writer.write(capture)

        # check keystroke to exit (image window must be on focus)
        key = cv.pollKey()
        if key == ord('q') or key == ord('Q') or key == 27:
            break


    cv.destroyAllWindows()
    if stream.isOpened():  stream.release()
    if writer.isOpened():  writer.release()
