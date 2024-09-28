#!/usr/bin/env python3
# encoding: utf-8

import os
import glob

import math
import numpy as np
import cv2 as cv
import cv2.aruco

import yaml
from yaml.loader import SafeLoader

from src.utils import projectCenter, interpolate_points, getMaskHue
from src.ArucoBoardHandler import ArucoBoardHandler

"""
    Class that handles the detection, and fixations of the probe panel shown to the participant
    with the target piece to look for
"""
class PanelHandler:
    def __init__(self, panel_configuration_path, colors_dict, colors_list, distortion_handler):
        
        self.distortion_handler = distortion_handler

        self.colors_list = colors_list
        self.colors_dict = colors_dict

        self.shape_contour = None
        self.last_detected = None
        self.homography = None

        self.panel_handler_list = self.parseCFGPanelData(panel_configuration_path)
    
    def step(self, image):
        undistorted_image = self.distortion_handler.undistortImage(image)
        
        self.panel_view = self.computeApplyHomography(undistorted_image)
        self.shape_contour = self.detectContour(self.panel_view)
        # self.panel_view = undistorted_image

    def computeApplyHomography(self, undistorted_image):
        
        self.last_detected = None
        for aruco_handler in self.panel_handler_list:
            homography, self.warp_width, self.warp_height = aruco_handler.getTransform(undistorted_image)
            if homography is not None:
                self.homography = homography
                self.last_detected = {'color': aruco_handler.color, 'shape': aruco_handler.shape}
                break

        display_image = np.zeros((self.warp_width, self.warp_height, 3), dtype=undistorted_image.dtype)
            
        if self.homography is not None and self.last_detected is not None:
            display_image = cv.warpPerspective(undistorted_image, self.homography, (self.warp_width, self.warp_height))

        return display_image

    def parseCFGPanelData(self, panel_configuration_path):

        panel_handler_list = []

        yaml_files = glob.glob(os.path.join(panel_configuration_path, '*.yaml')) + \
                    glob.glob(os.path.join(panel_configuration_path, '*.yml'))

        for yaml_file in yaml_files:
            shape = yaml_file.split('/')[-1].split('_')[0]
            color = yaml_file.split('/')[-1].split('_')[1].split('.')[0]
            panel_handler_list.append(ArucoBoardHandler(arucoboard_cfg_path=yaml_file, colors_list=self.colors_list, color=color, shape=shape))
        return panel_handler_list
            

    def handleVisualization(self, image, shape_contour):
        display_cfg_panel_view = image.copy()

        for panel_handler in self.panel_handler_list:
            ret = panel_handler.handleVisualization(display_cfg_panel_view)
            if ret:
                color = self.colors_list[panel_handler.color]
                cv.drawContours(display_cfg_panel_view, [shape_contour], -1, color=color, thickness=2)
        
        return display_cfg_panel_view

    def getVisualization(self):
        if self.last_detected is not None:
            return self.handleVisualization(self.panel_view, self.shape_contour)

        return self.panel_view
    
    
    def detectContour(self, image):
        shape_contour = None
        if self.last_detected is not None and image is not None:
            h_ref = self.colors_dict[self.last_detected['color']]

            hue, sat, intensity = cv.split(cv.cvtColor(image, cv.COLOR_BGR2HSV_FULL))
            res = getMaskHue(hue, sat, intensity, h_ref['h'], h_ref['eps'])

            edge_image = cv.Canny(res, threshold1=50, threshold2=200)
            contours, hierarchy = cv.findContours(edge_image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                perimeter = cv.arcLength(contour, True)
                if not perimeter > 0.1:
                    continue
                
                area = cv.contourArea(contour)
                if area < 1000 or area > math.inf:
                    continue
                
                shape_contour = cv.approxPolyDP(contour, .01 * perimeter, True)
                
        return shape_contour
    

    def getPixelInfo(self, coordinates):

        if coordinates is not None:
            pass
        shape, aruco, panel = False, False, False
        
        return shape, aruco, panel