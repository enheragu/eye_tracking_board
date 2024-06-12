#!/usr/bin/env python3
# encoding: utf-8

import os 
import json
import cv2 as cv
import numpy as np

cameraMatrix = None
distCoeffs = None

mapx = None
mapy = None
roi_undistort = None

homography = None
warpedWidth = None
warpedHeight = None

from src.perspective_correction import margin_four_point_transform


def initCalibrationData(frame_height, frame_width, calibration_json = 'camera_calib.json'):
    global cameraMatrix,distCoeffs
    global mapx,mapy,roi_undistort

    if os.path.exists(calibration_json):
        try:
            with open(calibration_json) as file:
                data = json.load(file)

            cameraMatrix = np.array(data['camera_matrix'])
            distCoeffs = np.array(data['distortion_coefficients'])

            print(f'Camera matrix: \n{cameraMatrix}')
            print(f'distortion_coefficients: \n{distCoeffs}')
        except:
            print("Calibration file not valid.")


    # If calibration data is available, undistort the image
    if cameraMatrix is not None:
        h, w = frame_height, frame_width
        newcameramtx, roi_undistort = cv.getOptimalNewCameraMatrix(cameraMatrix, distCoeffs, (w,h), alpha=0, newImgSize=(w,h))
        mapx, mapy = cv.initUndistortRectifyMap(cameraMatrix, distCoeffs, None, newcameramtx, (w,h), 5)
    

"""
    Corrects distortion and perspective
"""
def correctImage(capture):
    global homography, warpedWidth, warpedHeight
    global mapx,mapy,roi_undistort

    # If calibration data is available, undistort the image
    if cameraMatrix is not None:
        capture = cv.remap(capture, mapx, mapy, cv.INTER_LINEAR)

        # crop the image to given roi (alpha = 0 needs no roi?¿)
        x, y, w, h = roi_undistort
        capture = capture[y:y+h, x:x+w]

    # x0,y0 = 210,210
    # x1,y1 = 990,690
    # capture = capture[y0:y1, x0:x1]

    display_image = capture.copy()

    from src.detect_shapes import detectBoardContour

    board_contour = detectBoardContour(capture, None)
    if board_contour is not None:
        # homography, warpedWidth, warpedHeight = four_point_transform(board_contour.reshape(4, 2))
        homography, warpedWidth, warpedHeight = margin_four_point_transform(board_contour.reshape(4, 2), capture.shape)
        text = 'Distortion and Homography'
    else:
        text = 'Distortion only'

    if homography is not None:
        cv.putText(display_image, text, org=(10,30), fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0,255,0), thickness=2, lineType=cv.LINE_AA)
        capture = cv.warpPerspective(capture, homography, (warpedWidth, warpedHeight))
        display_image = cv.warpPerspective(display_image, homography, (warpedWidth, warpedHeight))
    
    return capture, display_image

"""
    Translates from original pixel coordinates to new image coordinates
    once distortion and perspective is corrected
"""
def correctCoordinates(original_coords):
    global homography, cameraMatrix, distCoeffs

    original_coord_np = np.array(original_coords, dtype=np.float32)
    undistorted_coord = cv.undistortPoints(original_coord_np, cameraMatrix, distCoeffs)
    # transformed_coord = cv.perspectiveTransform(undistorted_coord.reshape(-1, 1, 2), homography)

    return undistorted_coord

"""

"""
def reverseCoordinates(transformed_coords):
    global homography, cameraMatrix, distCoeffs
    
    transformed_coords = np.array(transformed_coords, dtype=np.float32)
    undistorted_coord = cv.perspectiveTransform(transformed_coords.reshape(-1, 1, 2), np.linalg.inv(homography))

    ptsOut = np.array(undistorted_coord, dtype='float32')
    ptsTemp = np.array([], dtype='float32')
    rtemp = ttemp = np.array([0,0,0], dtype='float32')
    ptsOut = cv.undistortPoints(ptsOut, cameraMatrix, None)
    ptsTemp = cv.convertPointsToHomogeneous( ptsOut )
    projected_points, _ = cv.projectPoints( ptsTemp, rtemp, ttemp, cameraMatrix, distCoeffs, ptsOut )

    return projected_points