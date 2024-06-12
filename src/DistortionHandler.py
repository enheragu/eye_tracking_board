#!/usr/bin/env python3
# encoding: utf-8

import os 
import json
import cv2 as cv
import numpy as np



from src.perspective_correction import margin_four_point_transform

class DistortionHandler():
    def __init__(self, calibration_json_path, frame_width, frame_height):
        self.cameraMatrix, self.distCoeffs = self.parseCalibrationData(calibration_json_path)
        self.newcameratx, self.roi_undistort, self.mapx, self.mapy = self.undistortImageParams(frame_height, frame_width)

    def parseCalibrationData(self, calibration_json_path = 'camera_calib.json'):
        if os.path.exists(calibration_json_path):
            try:
                with open(calibration_json_path) as file:
                    data = json.load(file)

                cameraMatrix = np.array(data['camera_matrix'])
                distCoeffs = np.array(data['distortion_coefficients'])

                print(f'Camera matrix: \n{cameraMatrix}')
                print(f'distortion_coefficients: \n{distCoeffs}')
            except:
                print("Calibration file not valid.")
        return cameraMatrix, distCoeffs

    def undistortImageParams(self, frame_height, frame_width):
        # If calibration data is available, undistort the image
        if self.cameraMatrix is not None:
            h, w = frame_height, frame_width
            newcameramtx, roi_undistort = cv.getOptimalNewCameraMatrix(self.cameraMatrix, self.distCoeffs, (w,h), alpha=0, newImgSize=(w,h))
            mapx, mapy = cv.initUndistortRectifyMap(self.cameraMatrix, self.distCoeffs, None, newcameramtx, (w,h), 5)
            return newcameramtx, roi_undistort, mapx, mapy

    """
        Corrects distortion and perspective
    """
    def undistortImage(self,capture):
        global mapx,mapy,roi_undistort

        display_image = capture.copy()

        # If calibration data is available, undistort the image
        if self.cameraMatrix is not None:
            display_image = cv.remap(display_image, self.mapx, self.mapy, cv.INTER_LINEAR)
            # crop the image to given roi (alpha = 0 needs no roi?¿)
            x, y, w, h = self.roi_undistort
            display_image = display_image[y:y+h, x:x+w]

        return display_image

    """
        Translates from original pixel coordinates to new image coordinates
        once distortion and perspective is corrected.
        An homography can be provided to correct it along with the distortion.
    """
    def correctCoordinates(self, original_coords, homography = None):
        
        original_coord_np = np.array(original_coords, dtype=np.float32)
        dst = cv.undistortPoints(original_coord_np, self.cameraMatrix, self.distCoeffs)
        if homography is not None:
            dst = cv.perspectiveTransform(dst.reshape(-1, 1, 2), homography)

        return dst

    """
        Translates from new_image coordinates to undistorted image.
        An homography can be provided to correct it along with the distortion.
    """
    def reverseCoordinates(self, transformed_coords, homography = None):
        
        dst = np.array(transformed_coords, dtype=np.float32)
        if homography is not None:
            dst = cv.perspectiveTransform(transformed_coords.reshape(-1, 1, 2), np.linalg.inv(homography))

        ptsOut = np.array(dst, dtype='float32')
        ptsTemp = np.array([], dtype='float32')
        rtemp = ttemp = np.array([0,0,0], dtype='float32')
        ptsOut = cv.undistortPoints(ptsOut, self.cameraMatrix, None)
        ptsTemp = cv.convertPointsToHomogeneous( ptsOut )
        dst, _ = cv.projectPoints( ptsTemp, rtemp, ttemp, self.cameraMatrix, self.distCoeffs, ptsOut )

        return dst