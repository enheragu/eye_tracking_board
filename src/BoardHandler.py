
#!/usr/bin/env python3
# encoding: utf-8

import math

import cv2 as cv
import numpy as np

import yaml
from yaml.loader import SafeLoader

from src.ArucoBoardHandler import ArucoBoardHandler, aruco_board_transform

from src.utils import interpolate_points, getMaskHue
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

    def __init__(self, aruco_board_cfg_path, game_cfg_path, colors_dict, colors_list, distortion_handler):

        self.distortion_handler = distortion_handler

        self.colors_dict = colors_dict
        self.colors_list = colors_list
        self.color = self.colors_list['board']
        self.aruco_board_handler = ArucoBoardHandler(arucoboard_cfg_path=aruco_board_cfg_path, color='board', 
                                                     colors_list=self.colors_list, 
                                                     cameraMatrix=self.distortion_handler.cameraMatrix, 
                                                     distCoeffs=self.distortion_handler.distCoeffs)
        self.board_size, self.board_size_mm, self.board_data_dict_upright, self.board_data_dict_rotated = self.parseCFGBoardData(game_cfg_path)
        self.board_data_dict = self.board_data_dict_upright

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

        self.cell_matrix, self.cell_width, self.cell_height = None, None, None

        ## Add inertia to board contour, if not found in N frames just propagates
        # previous
        self.prev_board_contour = None
        self.board_contour_inertia_step = 2
        self.board_contour_intertia_counter = 0
        
        
    def step(self, image):
        undistorted_image = self.distortion_handler.undistortImage(image)
        self.board_view = self.computeApplyHomography(undistorted_image)
        self.board_contour = self.detectContour(self.board_view)

        if self.board_contour is None:
            if self.board_contour_inertia_step > self.board_contour_intertia_counter:
                self.board_contour = self.prev_board_contour
                self.board_contour_intertia_counter+=1
            else:
                self.prev_board_contour = None
                self.board_contour_intertia_counter = 0
        else:
            self.board_contour_intertia_counter = 0
            self.prev_board_contour = self.board_contour

        
        self.cell_matrix, self.cell_width, self.cell_height = None, None, None
        if self.board_contour is not None and len(self.board_contour) != 0:
            self.cell_matrix, self.cell_width, self.cell_height = self.computeBoardMatrix(self.board_contour, self.board_size)
            self.board_data_dict = self.completeBoardConfig(self.cell_matrix, self.cell_width, self.cell_height, self.board_data_dict)
        

    def handleVisualization(self, image, board_contour, board_size, cell_matrix, cell_contours, board_data_dict, fixation_coord):
        display_cfg_board_view = image.copy()
        display_detected_board_view = image.copy()

        if self.display_detected_board_contour and board_contour is not None:
            cv.drawContours(display_detected_board_view, [board_contour], -1, color=self.color, thickness=2)

        if self.display_configuration_board_matrix and cell_matrix is not None \
           and cell_contours is not None:
            cv.drawContours(display_cfg_board_view, cell_contours, -1, color=self.color, thickness=2)
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
    
    def getVisualization(self, original_image):
        if self.board_view is None:
            return original_image, original_image
        
        return self.handleVisualization(
            image=self.board_view,
            board_contour=self.board_contour, 
            board_size=self.board_size, 
            cell_matrix=self.cell_matrix, 
            cell_contours=self.cell_contours, 
            board_data_dict=self.board_data_dict,
            fixation_coord=self.fixation_coord)

    def getDistortedOriginalVisualization(self, undistorted_image):
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

    def computeApplyHomography(self, undistorted_image):
        ## Version detecting contour of black edges of the game
        display_image = undistorted_image
        # board_contour = self.detectContour(undistorted_image)
        # if board_contour is not None: 
            # Uses detected board contour as if it were an aruco
            # board_image_contours = np.array([board_contour.reshape(4, 2)], dtype=np.float32)
            # self.homography, self.warpedWidth, self.warpedHeight = aruco_board_transform(
            #                         aruco_image_contours=board_image_contours,
            #                         aruco_3d_contours=self.board_corners_3d,
            #                         board_3d_contours=self.margin_points_3d,
            #                         img_shape=undistorted_image.shape)

        self.homography, self.warp_width, self.warp_height, rotated = self.aruco_board_handler.getTransform(undistorted_image)
        self.board_data_dict = self.board_data_dict_rotated if rotated else self.board_data_dict_upright 

        # display_image = np.zeros((self.warp_width, self.warp_height, 3), dtype=undistorted_image.dtype)
        
        if self.homography is not None:
            display_image = cv.warpPerspective(undistorted_image, self.homography, (self.warpedWidth, self.warpedHeight))

        return display_image

    """
        Gets image coordinates and returns info about where in the board was it located
        (if its in the board at all)
    """
    def getPixelInfo(self, coordinates):
        self.fixation_coord = None
        if coordinates is not None and self.board_contour is not None and len(self.board_contour) != 0 \
            and self.board_data_dict is not None:
            
            self.fixation_coord = self.distortion_handler.correctCoordinates(coordinates, self.homography)
            # print(f'Original coordinates: {coordinates = }')
            # print(f'Fixation projected: {self.fixation_coord = }')
            idx = self.getCellIndex(self.fixation_coord)
            
            if idx[0] is not None:
                color = self.board_data_dict[idx][0]
                shape = self.board_data_dict[idx][1]
                slot = self.board_data_dict[idx][2]
                board_coord = idx

                # print(f"Fixation detected in: {self.board_data_dict[idx]} in {board_coord}")
                return color, shape, slot, board_coord
        
        return None, None, None, None

    ## FUNCTIONS BASED ON CONFIGURATION
    def parseCFGBoardData(self,game_configuration):
        board_data_dict = {}
        with open(game_configuration) as file:
            data = yaml.load(file, Loader=SafeLoader)
            board_data_dict_upright = {tuple(map(int, key.split(','))): value for key, value in data['board_config'].items()}
            board_size = data['board_size']      
            board_size_mm = data['board_size_mm']

            margin = 10 # in mm, same dimension as board_size
            self.board_corners_3d = np.array([[
                [margin, margin, 0],                      # Esquina superior izquierda
                [margin, board_size_mm[1]+margin, 0],           # Esquina inferior izquierda
                [board_size_mm[0]+margin, margin, 0],           # Esquina superior derecha
                [board_size_mm[0]+margin, board_size_mm[1]+margin, 0] # Esquina inferior derecha
            ]], dtype=np.float32)
            

            self.margin_points_3d = np.array([[
                [0, 0, 0],                                        # Esquina superior izquierda
                [0, board_size_mm[1]+margin*2, 0],                    # Esquina inferior izquierda
                [board_size_mm[0]+margin*2, 0, 0],                     # Esquina superior derecha
                [board_size_mm[0]+margin*2, board_size_mm[1]+margin*2, 0]  # Esquina inferior derecha
            ]], dtype=np.float32)

        def rotate_coordinates_180(coords, board_size):
            return (board_size[0] - coords[0] - 1, board_size[1] - coords[1] - 1)
        board_data_dict_rotated = {rotate_coordinates_180(key, board_size): value for key, value in board_data_dict_upright.items()}

        return board_size, board_size_mm, board_data_dict_upright, board_data_dict_rotated
    
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
        x, y = pixel_coord[0]
        
        x_board_origin = self.cell_matrix[0][0][0] - self.cell_width / 2
        y_board_origin = self.cell_matrix[0][0][1] - self.cell_height / 2

        relative_x = x - x_board_origin
        relative_y = y - y_board_origin

        cell_row = int(relative_y // self.cell_height)
        cell_col = int(relative_x // self.cell_width)
        
        if 0 <= cell_row < self.cell_matrix.shape[0] and \
        0 <= cell_col < self.cell_matrix.shape[1]:
            return int(cell_col), int(cell_row)
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
        hsv_image = cv.cvtColor(image, cv.COLOR_BGR2HSV_FULL)
        hue, sat, intensity = cv.split(hsv_image)

        # Image is already compensated, get color infomration from board margins in the image
        # instead of harcoding them:
        height, width, _ = image.shape
        pixel_distance = 15
        reference_edges = np.concatenate([
            hsv_image[pixel_distance:height-pixel_distance, pixel_distance],  # Left edge
            hsv_image[pixel_distance:height-pixel_distance, width-pixel_distance],  # Right edge
            hsv_image[pixel_distance, pixel_distance:width-pixel_distance],  # Top edge
            hsv_image[height-pixel_distance, pixel_distance:width-pixel_distance]  # Bottom edge
        ])
        
        ### Logging reference
        # if reference_edges.size > 0:
        #     image_reference_edges = image.copy()
            
        #     for y in range(pixel_distance, height-pixel_distance):
        #         cv.circle(image_reference_edges, (pixel_distance, y), 1, (0, 255, 0), -1)  # Point (x=4, y=y)
        #         cv.circle(image_reference_edges, (width-pixel_distance, y), 1, (0, 255, 0), -1)  # Point (x=width-4, y=y)

        #     for x in range(pixel_distance, width-pixel_distance):
        #         cv.circle(image_reference_edges, (x, pixel_distance), 1, (0, 255, 0), -1)  # Point (x=x, y=4)
        #         cv.circle(image_reference_edges, (x, height-pixel_distance), 1, (0, 255, 0), -1)  # Point (x=x, y=height-4)

        #     cv.imshow(f'reference_edges', image_reference_edges)

        mean_h, mean_s, mean_v = np.mean(reference_edges, axis=0)
        std_h, std_s, std_v = np.std(reference_edges, axis=0)
        s_margins = [min(0, mean_s-4*std_s), min(255, mean_s+4*std_s)]
        v_margins = [min(0, mean_v-4*std_v), min(255, mean_v+4*std_v)]

        # print(f"{mean_h = }\t{mean_s = }\t{mean_v = }")
        # print(f"{std_h = }\t{std_s = }\t{std_v = }")
        
        # Take any hue with low brightness
        # res = getMaskHue(hue, sat, intensity, h_ref=0, h_epsilon=180, s_margins=[0,255], v_margins = [0,90])
        res = getMaskHue(hue, sat, intensity, h_ref=mean_h, h_epsilon=4*std_h, s_margins=s_margins, v_margins = v_margins)
        # cv.imshow(f'border_mask', res)

        edge_image = cv.Canny(res, threshold1=50, threshold2=200)
        contours, hierarchy = cv.findContours(edge_image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        
        board_contour = None
        # cv.imshow(f'border_edges', edge_image)
        for contour in contours:
            shape, approx = checkShape(contour, area_filter=[700000, math.inf])
            if shape == 'rectangle':
                board_contour = approx
                area = cv.contourArea(contour)
                break
 
        # cv.imshow(f'border_edges', edge_image)
        # cv.imshow(f'border_contour', display_image)
        return board_contour
    
    def isContourDetected(self):
        return True if self.board_contour is not None else False

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
