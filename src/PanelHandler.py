#!/usr/bin/env python3
# encoding: utf-8

import math
import numpy as np
import cv2 as cv
import cv2.aruco

import yaml
from yaml.loader import SafeLoader

from src.utils import projectCenter, interpolate_points, getMaskHue
from src.perspective_correction import aruco_board_transform, rescale_3d_points

"""
    Class that handles the detection, and fixations of the probe panel shown to the participant
    with the target piece to look for
"""
class PanelHandler:
    def __init__(self, panel_configuration_path, colors_dict, colors_list, distortion_handler):
        
        self.distortion_handler = distortion_handler
        self.panel_data_dict = self.parseCFGPanelData(panel_configuration_path)

        self.aruco_dictionary = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_ARUCO_MIP_36H12)

        self.colors_list = colors_list
        self.colors_dict = colors_dict
        self.detected_aruco = None
        self.sample_contour = None

        self.homography = None
    
    def step(self, image):
        undistorted_image = self.distortion_handler.undistortImage(image)

        gray_image = cv.cvtColor(undistorted_image, cv.COLOR_BGR2GRAY)  # transforms to gray level
        corners, ids, rejectedImgPoints = cv.aruco.detectMarkers(gray_image, self.aruco_dictionary)

        self.detected_aruco = None
        if ids is not None and ids.size > 0:
            for aruco_data in self.panel_data_dict:
                index_array = np.where(ids == aruco_data['id'])
                if index_array[0].size > 0:
                    index = index_array[0][0]
                    self.detected_aruco = {'data': aruco_data, 'contour': corners[index], 'id': ids[index][0]}
                    break
        
        self.panel_view = self.computeApplyHomography(undistorted_image)
        self.sample_contour = self.detectContour(self.panel_view)
        # self.panel_view = undistorted_image

    def computeApplyHomography(self, image):
        
        _, new_shape = rescale_3d_points(self.sheet_points_3d[0], image.shape)
        display_image = np.zeros(new_shape, dtype=image.dtype)
    
        if self.detected_aruco is not None:
            self.homography, self.warp_width, self.warp_height = aruco_board_transform(
                                    aruco_image_contours=self.detected_aruco['contour'],
                                    aruco_3d_contours=self.aruco_corners_3d,
                                    board_3d_contours=self.sheet_points_3d,
                                    img_shape=image.shape)
            display_image = cv.warpPerspective(image, self.homography, (self.warp_width, self.warp_height))
            
        if self.homography is not None and self.detected_aruco is not None:
            display_image = cv.warpPerspective(image, self.homography, (self.warp_width, self.warp_height))

        return display_image

    def parseCFGPanelData(self,panel_configuration_path):
        panel_data_dict = {}
        with open(panel_configuration_path) as file:
            data = yaml.load(file, Loader=SafeLoader)
            panel_data_dict = data['marker_configuration']
            sample_sheet_size = data['sample_sheet_size']
            aruco_position = data['aruco_position']
            aruco_side = data['aruco_side']

            self.aruco_corners_3d = np.array([[
                [aruco_position[0], aruco_position[1]],                      # Esquina superior izquierda
                [aruco_position[0], aruco_position[1]+aruco_side],           # Esquina inferior izquierda
                [aruco_position[0]+aruco_side, aruco_position[1]],           # Esquina superior derecha
                [aruco_position[0]+aruco_side, aruco_position[1]+aruco_side] # Esquina inferior derecha
            ]], dtype=np.float32)
            

            self.sheet_points_3d = np.array([[
                [0, 0],                                        # Esquina superior izquierda
                [0, sample_sheet_size[1]],                    # Esquina inferior izquierda
                [sample_sheet_size[0], 0],                     # Esquina superior derecha
                [sample_sheet_size[0], sample_sheet_size[1]]  # Esquina inferior derecha
            ]], dtype=np.float32)

        
        return panel_data_dict

    def handleVisualization(self, image, aruco_data, sample_contour):
        display_cfg_panel_view = image.copy()

        corners, ids, rejectedImgPoints = cv.aruco.detectMarkers(display_cfg_panel_view, self.aruco_dictionary)
        
        cv.aruco.drawDetectedMarkers(display_cfg_panel_view, corners, ids, borderColor=(0,0,255))

        for index, aruco in enumerate(corners):
            aruco_data_current = None
            for aruco_data in self.panel_data_dict:
                index_array = np.where(ids == aruco_data['id'])
                if index_array[0].size > 0:
                    index = index_array[0][0]
                    aruco_data_current = aruco_data
                    
            if aruco_data_current is not None:
                center = projectCenter(aruco)
                x, y, width, height = cv.boundingRect(aruco)

                text = f"Search for {aruco_data_current['color']} {aruco_data_current['shape']}"
                color = self.colors_list[aruco_data_current['color']]
                font = cv.FONT_HERSHEY_SIMPLEX
                scale = 1
                thickness = 1
                text_size, _ = cv.getTextSize(f'{text}', font, scale, thickness)
                text_origin = (int(center[0]-text_size[0]/2),int(center[1]+text_size[1]+3+height/2))
                cv.putText(display_cfg_panel_view, text, org=text_origin, fontFace=font, fontScale=scale, color=color, thickness=thickness, lineType=cv.LINE_AA)
        
                cv.drawContours(display_cfg_panel_view, [sample_contour], -1, color=color, thickness=2)
        # dx = aruco_corners[0][1][0] - aruco_corners[0][0][0]
        # dy = aruco_corners[0][1][1] - aruco_corners[0][0][1]
        # angle = np.degrees(np.arctan2(dy, dx))  
        # draw_rotated_text(display_cfg_panel_view, text, (center), angle)
        return display_cfg_panel_view

    def getVisualization(self):
        if self.detected_aruco is not None:
            return self.handleVisualization(self.panel_view, self.panel_data_dict, self.sample_contour)

        return self.panel_view
    
    
    def detectContour(self, image):
        sample_contour = None
        if self.detected_aruco is not None and image is not None:
            h_ref = self.colors_dict[self.detected_aruco['data']['color']]

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
                
                sample_contour = cv.approxPolyDP(contour, .01 * perimeter, True)
                
        return sample_contour