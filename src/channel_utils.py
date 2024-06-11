#!/usr/bin/env python3
# encoding: utf-8

import cv2 as cv
import numpy as np

def claheEqualization(channel):
    clahe = cv.createCLAHE(clipLimit=4.0, tileGridSize=(8,8))
    return clahe.apply(channel)

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
    
    # res = cv.erode(res,kernel, iterations = 2)
    res = cv.morphologyEx(res, cv.MORPH_OPEN, kernel,  iterations=2)
    res = cv.morphologyEx(res, cv.MORPH_CLOSE, kernel, iterations=2)
    
    return res