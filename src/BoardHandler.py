
#!/usr/bin/env python3
# encoding: utf-8

import math

import cv2 as cv
import numpy as np

import yaml
from yaml.loader import SafeLoader

from src.perspective_correction import margin_four_point_transform
from src.utils import interpolate_points, getMaskHue, claheEqualization
from src.detect_shapes import checkShape

def getCellWH(cell_matrix, i, j):
    x, y = cell_matrix[i, j]

    if j < cell_matrix.shape[1] - 1:
        w = cell_matrix[i, j + 1][0] - cell_matrix[i, j][0]
    elif j > 0:
        w = cell_matrix[i, j][0] - cell_matrix[i, j - 1][0]
    else:
        w = 0 

    if i < cell_matrix.shape[0] - 1:
        h = cell_matrix[i + 1, j][1] - cell_matrix[i, j][1]
    elif i > 0:
        h = cell_matrix[i, j][1] - cell_matrix[i - 1, j][1]
    else:
        h = 0  

    return w, h

"""
    Class that handles all the stuff needed to interface with the board game
"""
class BoardHandler:

    def __init__(self, board_cfg_path, colors_dict, colors_list, distortion_handler):

        self.distortion_handler = distortion_handler

        self.colors_dict = colors_dict
        self.colors_list = colors_list
        self.board_size, self.board_size_mm, self.board_data_dict = self.parseCFGBoardData(board_cfg_path)

        self.undistorted_image = None
        self.board_view = None
        self.display_cfg_board_view = None
        self.display_detected_board_view = None

        self.homography = None
        self.warpedWidth = None
        self.warpedHeight = None

        self.display_detected_board_contour = True      # Display detected contour of the board
        self.display_configuration_board_matrix = True  # Display matrix from configuration
        self.display_configuration_slots_info = True    # Display slot info from configuration
        self.display_fixation = True                    # Display user fixation in the board

        self.board_contour = None
        self.fixation_coord = None
        self.cell_contours = None
        
        
    def step(self, image):
        undistorted_image = self.distortion_handler.undistortImage(image)
        self.board_view = self.computeApplyHomography(undistorted_image)
        self.board_contour = self.detectContour(self.board_view)
        
        self.cell_matrix, self.cell_width, self.cell_height = None, None, None
        if self.board_contour is not None and len(self.board_contour) != 0:
            self.cell_matrix, self.cell_width, self.cell_height = self.computeBoardMatrix(self.board_contour, self.board_size)
            self.board_data_dict = self.completeBoardConfig(self.cell_matrix, self.cell_width, self.cell_height, self.board_data_dict)

    def handleVisualization(self, image, board_contour, board_size, cell_matrix, cell_contours, board_data_dict, fixation_coord):
        display_cfg_board_view = image.copy()
        display_detected_board_view = image.copy()

        if self.display_detected_board_contour and board_contour is not None:
            cv.drawContours(display_detected_board_view, [board_contour], -1, color=(255,255,0), thickness=2)

        if self.display_configuration_board_matrix and cell_matrix is not None \
           and cell_contours is not None:
            cv.drawContours(display_cfg_board_view, cell_contours, -1, color=(255,255,0), thickness=2)
            for i in range(board_size[1]):
                for j in range(board_size[0]):
                    x, y = cell_matrix[i, j]
                    w,h = getCellWH(cell_matrix=cell_matrix, i=i, j=j)
                    # print(f'[{i},{j}] in {(x+5-w/2,y+12-h/2)}')
                    cv.putText(display_cfg_board_view, f'[{i},{j}]', org=(int(x+5-w/2),int(y+12-h/2)), fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale=0.3, color=(0, 255, 0), thickness=1, lineType=cv.LINE_AA)
        
        if self.display_configuration_slots_info and board_data_dict is not None \
           and cell_matrix is not None:
            for i in range(len(cell_matrix)):
                for j in range(len(cell_matrix[i])):
                        data = board_data_dict[(j,i)]
                        center = cell_matrix[i][j]
                        text = f"{'Shape' if data[2] else 'Slot'}"
                        text2 = f"{data[1]}"
                        color = self.colors_list[data[0]]
                        font = cv.FONT_HERSHEY_SIMPLEX
                        scale = 0.5
                        thickness = 1
                        text_size, _ = cv.getTextSize(f'{text}', font, scale, thickness)
                        text_size2, _ = cv.getTextSize(f'{text2}', font, scale, thickness)
                        text_origin = (int(center[0]-text_size[0]/2),int(center[1]-text_size[1]+3))
                        text_origin2 = (int(center[0]-text_size2[0]/2),int(center[1]+text_size2[1]+3))
                        cv.putText(display_cfg_board_view, text, org=text_origin, fontFace=font, fontScale=scale, color=color, thickness=thickness, lineType=cv.LINE_AA)
                        cv.putText(display_cfg_board_view, text2, org=text_origin2, fontFace=font, fontScale=scale, color=color, thickness=thickness, lineType=cv.LINE_AA)
                        cv.circle(display_cfg_board_view, center, radius=3, color=color, thickness=-1)
        
        if self.display_fixation and fixation_coord is not None:
            fixation_coord_tuple = (int(fixation_coord[0][0]), int(fixation_coord[0][1]))
            cv.circle(display_cfg_board_view, fixation_coord_tuple, radius=10, color=(0,0,255), thickness=-1)
            cv.circle(display_detected_board_view, fixation_coord_tuple, radius=10, color=(0,0,255), thickness=-1)
            fixation_coord = None      # Reset fixation to be updated throguh function
    
        return display_cfg_board_view, display_detected_board_view
    
    def getVisualization(self):
        return self.handleVisualization(
            image=self.board_view,
            board_contour=self.board_contour, 
            board_size=self.board_size, 
            cell_matrix=self.cell_matrix, 
            cell_contours=self.cell_contours, 
            board_data_dict=self.board_data_dict,
            fixation_coord=self.fixation_coord)

    def getUndistortedVisualization(self, undistorted_image):
        if self.board_contour is not None:
            board_contour_extended = interpolate_points(self.board_contour).astype(np.float32)
            undistorted_board_contour = self.distortion_handler.reverseCoordinates(board_contour_extended, homography = self.homography).astype(np.int32)

            undistorted_fixation_coord = self.fixation_coord
            if undistorted_fixation_coord is not None:
                undistorted_fixation_coord = self.fixation_coord.astype(np.float32)
                undistorted_fixation_coord = self.distortion_handler.reverseCoordinates(undistorted_fixation_coord, homography = self.homography).astype(np.int32)[0]
            
            undistorted_cell_matrix = self.cell_matrix
            if undistorted_cell_matrix is not None:
                undistorted_cell_matrix = np.zeros((self.board_size[1], self.board_size[0], 2), dtype=int)

                for i in range(len(undistorted_cell_matrix)):
                    for j in range(len(undistorted_cell_matrix[i])):
                        coord = np.array([self.cell_matrix[i][j]], dtype=np.float32)
                        undistorted_cell_matrix[i][j] = self.distortion_handler.reverseCoordinates(coord, homography = self.homography).astype(np.int32)[0]

            undistorted_cell_contours = self.cell_contours
            if undistorted_cell_contours is not None:
                for index, contour in enumerate(undistorted_cell_contours):
                    contour_extended = interpolate_points(contour).astype(np.float32)
                    undistorted_cell_contours[index] = self.distortion_handler.reverseCoordinates(contour_extended, homography = self.homography).astype(np.int32)

            return self.handleVisualization(undistorted_image, undistorted_board_contour, self.board_size, undistorted_cell_matrix, undistorted_cell_contours, self.board_data_dict, undistorted_fixation_coord)

        # If cannot plot any data...
        return undistorted_image, undistorted_image

    def computeApplyHomography(self, image):
        board_contour = self.detectContour(image)
        if board_contour is not None:
            # homography, warpedWidth, warpedHeight = four_point_transform(board_contour.reshape(4, 2))
            self.homography, self.warpedWidth, self.warpedHeight = margin_four_point_transform(board_contour.reshape(4, 2), image.shape)
            
        if self.homography is not None:
            display_image = cv.warpPerspective(image, self.homography, (self.warpedWidth, self.warpedHeight))

        return display_image

    """
        Gets image coordinates and returns info about where in the board was it located
        (if its in the board at all)
    """
    def getPixelInfo(self, coordinates):
        if self.board_contour is not None and len(self.board_contour) != 0 \
            and self.board_data_dict is not None:
            idx = self.getCellIndex(coordinates[0])
            
            if idx[0] is not None:
                print(f"Fixation detected in: {self.board_data_dict[idx]}")
                color = self.board_data_dict[idx][0]
                shape = self.board_data_dict[idx][1]
                slot = self.board_data_dict[idx][2]

                self.fixation_coord = coordinates
                return color, shape, slot
        
        self.fixation_coord = None
        return None, None, None

    ## FUNCTIONS BASED ON CONFIGURATION
    def parseCFGBoardData(self,board_configuration):
        board_data_dict = {}
        with open(board_configuration) as file:
            data = yaml.load(file, Loader=SafeLoader)
            board_data_dict = {tuple(map(int, key.split(','))): value for key, value in data['board_config'].items()}
            board_size = data['board_size']      
            board_size_mm = data['board_size_mm']
        
        return board_size, board_size_mm, board_data_dict
    
    def computeBoardMatrix(self, board_contour, board_size):
        x, y, w, h = cv.boundingRect(board_contour)

        cell_width = w // board_size[0]
        cell_height = h // board_size[1]

        self.cell_contours = []
        cell_matrix = np.zeros((board_size[1], board_size[0], 2), dtype=int)
        for i in range(board_size[1]):
            for j in range(board_size[0]):
                x_center = x + j * cell_width + cell_width/2
                y_center = y + i * cell_height + cell_height/2
                
                cell_matrix[i,j] = [int(x_center),int(y_center)]

                top_left = (int(x_center - cell_width / 2), int(y_center - cell_height / 2))
                top_right = (int(x_center + cell_width / 2), int(y_center - cell_height / 2))
                bottom_right = (int(x_center + cell_width / 2), int(y_center + cell_height / 2))
                bottom_left = (int(x_center - cell_width / 2), int(y_center + cell_height / 2))
                self.cell_contours.append(np.array([[top_left], [top_right], [bottom_right], [bottom_left]]))

        return cell_matrix, cell_width, cell_height
    
    def getCellIndex(self, pixel_coord):
        x, y = pixel_coord
        
        cell_row = int((y - self.cell_matrix[0][0][1] + self.cell_height) // self.cell_height)
        cell_col = int((x - self.cell_matrix[0][0][0] + self.cell_width) // self.cell_width)
        
        if 0 <= cell_row < self.cell_matrix.shape[1] and \
        0 <= cell_col < self.cell_matrix.shape[0]:
            return int(cell_row), int(cell_col)
        else:
            return None, None
    
    def completeBoardConfig(self, cell_matrix, cell_width, cell_height, board_data_dict):
        if board_data_dict is not None:
            board_data_dict['cell_width'] = cell_width
            board_data_dict['cell_height'] = cell_height

            for i in range(len(cell_matrix)):
                for j in range(len(cell_matrix[i])):

                    data = board_data_dict[(j,i)]
                    x,y = cell_matrix[i][j]
                    
                    if len(board_data_dict[(j,i)]) <= 3:
                        board_data_dict[(j,i)].append(cell_matrix[i][j])
                    else:
                        board_data_dict[(j,i)][3] = cell_matrix[i][j]

        return board_data_dict

    ## FUNCTIONS BASED ON DETECTION OVER IMAGE
    def detectContour(self, image):
        hue, sat, intensity = cv.split(cv.cvtColor(image, cv.COLOR_BGR2HSV_FULL))
        # Take any hue with low brightness
        res = getMaskHue(hue, sat, intensity, h_ref=0, h_epsilon=180, s_margins=[0,255], v_margins = [0,70])
        # cv.imshow(f'border_mask', res)

        edge_image = cv.Canny(res, threshold1=50, threshold2=200)
        contours, hierarchy = cv.findContours(edge_image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        
        board_contour = None
        # cv.imshow(f'border_edges', edge_image)
        for contour in contours:
            shape, approx = checkShape(contour, area_filter=[10000, math.inf])
            if shape == 'rectangle':
                board_contour = approx
                break
                
        # cv.imshow(f'border_edges', edge_image)
        # cv.imshow(f'border_contour', display_image)
        return board_contour
    
    def detectSlots(self, image):
        pass
        ## Detection of squares and slots :)
        # might be unnecesary
        # detected_board_data = []
        # contours_dict = detectColorSquares(capture, color_dict=color_dict, colors_list=colors_list, display_image=None)
        # for color, contour_list in contours_dict.items():
        #     for contour in contour_list:
        #         # print(f'Detect shape for {color} and contour {contour}')
        #         is_slot, center = isSlot(capture, contour, color_dict[color], colors_list[color], display_image=None)
        #         detected_board_data.append([is_slot, center])
