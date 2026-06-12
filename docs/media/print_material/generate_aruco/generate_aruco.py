#!/usr/bin/env python3
# encoding: utf-8

import numpy as np
import argparse
import cv2 as cv
import sys

#############
# Configure how many markers are needed and its size in pixels
N_MARKERS = 30
MARKER_SIZE = 300
#############


if __name__ == '__main__':
    DICT_ARUCO_ORIGINAL_MAX = 1024
    aruco_dict = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_ARUCO_ORIGINAL)

    for marker_id in range(0,DICT_ARUCO_ORIGINAL_MAX,int(round(DICT_ARUCO_ORIGINAL_MAX/N_MARKERS))):
        marker_img = aruco_dict.generateImageMarker(marker_id,  MARKER_SIZE)
        cv.imwrite(f'aruco_dict_original_{marker_id}.png', marker_img)
        cv.imshow("ArUCo Tag", marker_img)
        cv.waitKey(0)
