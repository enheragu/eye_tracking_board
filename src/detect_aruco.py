#!/usr/bin/env python3
# encoding: utf-8


import cv2 as cv
import cv2.aruco

def drawArucos(capture):
    dictionary = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_ARUCO_MIP_36H12)
    
    gray_image = cv.cvtColor(capture, cv.COLOR_BGR2GRAY)  # transforms to gray level
    corners, ids, rejectedImgPoints = cv.aruco.detectMarkers(gray_image, dictionary)
    dispimage = cv.aruco.drawDetectedMarkers(capture, corners, ids, borderColor=(0,0,255))

    return dispimage