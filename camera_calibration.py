#!/usr/bin/env python3
# encoding: utf-8

# Import libraries
import cv2 as cv
import numpy as np
import os
import json
from multiprocessing import Pool
from tqdm import tqdm

from src.utils import log

# Calibrate camera with stored image files (calibration pattern)
calibration_video_path = './calibration.mp4'
# Calibration doe snot process all frames from video but from step to step :)
step = 10

# Calibration pattern inner corners (cols Y-axis, rows X-axis)
patternSize = (8, 6)
squareSize = 30.0       # square pattern side in mm


if __name__ == "__main__":


    # 3D object points coordinates (x,y,z)
    objp3D = np.zeros((patternSize[1], patternSize[0], 3), np.float32)
    for x in range(patternSize[1]):
        for y in range(patternSize[0]):
            objp3D[x,y] = (x*squareSize, y*squareSize, 0)


    # From objp3D.shape = (6, 8, 3)
    # To objp3D.shape = (48, 3)
    objp3D = objp3D.reshape(-1, 3)   #  transform in a  row vector of (x,y,z) tuples

    window_name = 'Calibration board detected'
    cv.namedWindow(window_name, cv.WINDOW_NORMAL)
    cv.resizeWindow(window_name, 800, 600)  # Tamaño deseado de la ventana (ancho, alto)


    objpoints = []      # 3D point in real world space
    imgpoints = []      # 2D points in image plane
    num_patterns = 0    # number of detected patterns. We need at lest 3

        

    stream = cv.VideoCapture(calibration_video_path)
    if not stream.isOpened():
        log(f"Could not open video {calibration_video_path}")
        exit()

    total_frames = int(stream.get(cv.CAP_PROP_FRAME_COUNT))  # Total de frames si es un archivo de video


    for frame_count in range(0, total_frames, step):
        stream.set(cv.CAP_PROP_POS_FRAMES, frame_count)
        ret, image = stream.read()
        if not ret:
            log("Can't receive frame (stream end?). Exiting ...")
            break
        
        gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

        patternWasFound, corners = cv.findChessboardCorners(gray_image, patternSize)

        if patternWasFound:
            # log(f"Corners found in image: {file_path}")
            num_patterns += 1
            termCriteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 50, 0.001)
            corners = cv.cornerSubPix(gray_image, corners, winSize=(11,11), zeroZone=(-1,-1), criteria=termCriteria)

            # add points to image/3Dobjet lists
            objpoints.append(objp3D)
            imgpoints.append(corners)

            # Image is quite huge, draw automatic and by hand points so they can be seen better
            cv.drawChessboardCorners(image, patternSize, corners, patternWasFound)
            for corner in corners:
                x, y = corner.ravel()
                cv.circle(image, (int(x), int(y)), 3, (0, 255, 0), thickness=20)  # Dibuja un círculo en cada esquina
        
        else:
            log(f"No corners found in capture")
        
        text = f"{frame_count}/{total_frames}" if total_frames > 0 else f"{frame_count}/?"
        cv.putText(image, text, org=(10,30), fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0,0,255), thickness=2, lineType=cv.LINE_AA)
        
        cv.imshow(window_name, image)

        # check keystroke to exit (image window must be on focus)
        key = cv.pollKey()
        # key = cv.waitKey(0)
        if key == ord('q') or key == ord('Q') or key == 27:
            break


    if num_patterns >= 3:
        log(f'Detected {num_patterns} patterns. Computing calibration matrix.')
        # imageSize:  (cols, rows)
        termCriteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 50, 0.001)
        cameraMatrix = np.array([])
        distCoeffs = np.array([])
        rms, cameraMatrix, distCoeffs, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, imageSize=gray_image.shape[::-1],
                                                        cameraMatrix=None, distCoeffs=None, flags=0, criteria=termCriteria)
        log(f"Used {num_patterns} patterns for camera calibration")
        log(f"RMS reprojection error: {rms} pixels")

        log('Camera matrix:', cameraMatrix)
        log('distortion_coefficients:', distCoeffs)
        
        # store calibration data in a JSON file
        with open('camera_calib.json', 'w') as file:
            json.dump({'camera_matrix': cameraMatrix.tolist(),
                    'distortion_coefficients': distCoeffs.tolist()}, file)
            
    else:
        log('Not enough patterns detected.')

    cv.destroyAllWindows()