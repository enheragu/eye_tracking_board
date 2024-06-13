#!/usr/bin/env python3
# encoding: utf-8

import numpy as np
import cv2 as cv
import cv2.aruco

import yaml
from yaml.loader import SafeLoader

from src.utils import projectCenter, draw_rotated_text
from src.perspective_correction import sort_points_clockwise, four_point_transform

"""
    Class that handles the detection, and fixations of the probe panel shown to the participant
    with the target piece to look for
"""
class PanelHandler:
    def __init__(self, panel_configuration_path, colors_list, distortion_handler):
        
        self.distortion_handler = distortion_handler
        self.panel_data_dict = self.parseCFGPanelData(panel_configuration_path)

        self.aruco_dictionary = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_ARUCO_MIP_36H12)

        self.colors_list = colors_list
        self.detected_aruco = None

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
        # self.panel_view = undistorted_image

    def computeApplyHomography(self, image):

        display_image = np.zeros(image.shape)
    
        if self.detected_aruco is not None:
            aruco_corners_image = sort_points_clockwise(self.detected_aruco['contour'].reshape(4, 2))
            aruco_corners_image = np.array(aruco_corners_image, dtype=np.float32) 

            # M0 is transformation between corners in image and corrected persective image
            M0, _, _ = four_point_transform(aruco_corners_image, zero_coord=False)

            # Position of aruco in sheet coordinates -> findhomography between aruco in sheet and in image
            M1, _ = cv.findHomography(self.aruco_corners_3d, aruco_corners_image)

            # Position of sheet in sheet coordinates -> find how these points translate to image
            sheet_points_image = cv.perspectiveTransform(self.sheet_points_3d, M1)
            sheet_points_image = np.array([sort_points_clockwise(sheet_points_image[0])]).astype("float32")
            sheet_points_image = cv.perspectiveTransform(sheet_points_image, M0)
            sheet_points_image = sort_points_clockwise(sheet_points_image[0]).astype("float32")

            # findHomography between sheet and its computed projection in image
            # self.homography = cv.getPerspectiveTransform(sheet_points_3d, sheet_points_image)
            self.homography, _ = cv.findHomography(self.sheet_points_3d, sheet_points_image)

            self.homography = M0
            width, height = image.shape[1], image.shape[0]
            display_image = cv.warpPerspective(image, self.homography, (width, height))
            

        if self.homography is not None and self.detected_aruco is not None:
            width, height = image.shape[1], image.shape[0]
            display_image = cv.warpPerspective(image, self.homography, (width, height))

        return display_image

    def parseCFGPanelData(self,panel_configuration_path):
        panel_data_dict = {}
        with open(panel_configuration_path) as file:
            data = yaml.load(file, Loader=SafeLoader)
            panel_data_dict = data['marker_configuration']
            sample_sheet_size = data['sample_sheet_size']
            aruco_position = data['aruco_position']
            aruco_side = data['aruco_side']

            self.aruco_corners_3d = sort_points_clockwise(np.array([
                [aruco_position[0], aruco_position[1]],                      # Esquina superior izquierda
                [aruco_position[0], aruco_position[1]+aruco_side],           # Esquina inferior izquierda
                [aruco_position[0]+aruco_side, aruco_position[1]],           # Esquina superior derecha
                [aruco_position[0]+aruco_side, aruco_position[1]+aruco_side] # Esquina inferior derecha
            ], dtype=np.float32))
            
            # M1 is transformation between aruco in 3d and aruco in image
            self.aruco_corners_3d = np.array([self.aruco_corners_3d.astype("float32")])

            self.sheet_points_3d = sort_points_clockwise(np.array([
                [0, 0],                                        # Esquina superior izquierda
                [0, sample_sheet_size[1]],                    # Esquina inferior izquierda
                [sample_sheet_size[0], 0],                     # Esquina superior derecha
                [sample_sheet_size[0], sample_sheet_size[1]]  # Esquina inferior derecha
            ], dtype=np.float32))
            self.sheet_points_3d = np.array([self.sheet_points_3d.astype("float32")])
        
        return panel_data_dict

    def handleVisualization(self, image, aruco_data, aruco_corners, aruco_id):
        display_cfg_panel_view = image.copy()

        aruco_id = np.array([[self.detected_aruco['id']]])  # Convertir el id a array de NumPy 2D
        aruco_corners = np.array(self.detected_aruco['contour'])  # Asegurarse de que los corners son un array de NumPy
        cv.aruco.drawDetectedMarkers(display_cfg_panel_view, [aruco_corners], aruco_id, borderColor=(0,0,255))

        center = projectCenter(aruco_corners)
        x, y, width, height = cv.boundingRect(aruco_corners)

        text = f"Search for {aruco_data['color']} {aruco_data['shape']}"
        color = self.colors_list[aruco_data['color']]
        font = cv.FONT_HERSHEY_SIMPLEX
        scale = 1
        thickness = 1
        text_size, _ = cv.getTextSize(f'{text}', font, scale, thickness)
        text_origin = (int(center[0]-text_size[0]/2),int(center[1]+text_size[1]+3+height/2))
        cv.putText(display_cfg_panel_view, text, org=text_origin, fontFace=font, fontScale=scale, color=color, thickness=thickness, lineType=cv.LINE_AA)
                      
        # dx = aruco_corners[0][1][0] - aruco_corners[0][0][0]
        # dy = aruco_corners[0][1][1] - aruco_corners[0][0][1]
        # angle = np.degrees(np.arctan2(dy, dx))  
        # draw_rotated_text(display_cfg_panel_view, text, (center), angle)
        return display_cfg_panel_view

    def getVisualization(self):
        if self.detected_aruco is not None:
            return self.handleVisualization(self.panel_view, self.detected_aruco['data'], 
                                            self.detected_aruco['contour'], self.detected_aruco['id'])

        return self.panel_view