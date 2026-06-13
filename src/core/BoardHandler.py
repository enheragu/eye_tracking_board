
#!/usr/bin/env python3
# encoding: utf-8

import math
from collections import deque

import cv2 as cv
import numpy as np

import yaml
from yaml.loader import SafeLoader

from src.core.ArucoBoardHandler import ArucoBoardHandler

from src.core.utils import interpolate_points, getMaskHue, log, log_debug
from src.core.detect_shapes import checkShape

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
        # min_markers=4: enough for a stable homography (4 markers = 16 corner
        # points well spread) while tolerating the hand covering several markers
        # during the grab, down from the previous ~5 of 10
        self.aruco_board_handler = ArucoBoardHandler(arucoboard_cfg_path=aruco_board_cfg_path, color='board',
                                                     colors_list=self.colors_list,
                                                     cameraMatrix=self.distortion_handler.cameraMatrix,
                                                     distCoeffs=self.distortion_handler.distCoeffs,
                                                     min_markers=4)
        self.board_size, self.board_size_mm, self.board_data_dict_upright, self.board_data_dict_rotated = self.parseCFGBoardData(game_cfg_path)
        self.board_data_dict = self.board_data_dict_upright

        self.undistorted_image = None
        self.board_view = None
        self.display_cfg_board_view = None
        self.display_detected_board_view = None

        self.homography = None
        self.warp_width = None
        self.warp_height = None
        self.contour_detected_raw = False

        self.display_detected_board_contour = True      # Display detected contour of the board
        self.display_configuration_board_matrix = True  # Display matrix from configuration
        self.display_configuration_slots_info = True    # Display slot info from configuration
        self.display_fixation = True                    # Display user fixation in the board
        self.display_arucos = True                      # Display ArucosDetected in the view

        self.board_contour = None
        self.fixation_coord_list = []
        self.cell_contours = None

        self.cell_matrix, self.cell_width, self.cell_height = None, None, None
        self.board_origin, self.board_width, self.board_height = None, None, None

        ## Add inertia to board contour, if not found in N frames just propagates
        # previous
        self.prev_board_contour = None
        self.board_contour_inertia_step = 2
        self.board_contour_intertia_counter = 0

        ## Session reference of the board position inside the warped view. The warp
        # is built from the aruco layout, so the border rectangle lands in the same
        # warp coordinates the whole session: the median of past detections allows
        # assigning gaze to cells when the homography is valid but the full border
        # is not detectable (panel removal window, partial occlusions)
        self.reference_rect_history = deque(maxlen=25)
        self.reference_board_rect = None
        self.grid_from_reference = False

        ## Target area occlusion tracking (trial end detection). Temporal change
        # detection against a reference patch captured at trial start: works whatever
        # the hand/sleeve color is, and the relief/shadows of the 3D pieces are baked
        # into the reference (their variation is global and discounted by the control
        # cells). Double-margin: the target patch is split into its white pixels
        # (gutters/borders) and its colored pixels, and occlusion is confirmed only
        # when BOTH change (min of the two), so a partial shadow on one does not fire.
        self.target_cell = None
        self.control_cells = []
        self.tracking_ref_patches = None
        self.tracking_ref_masks = None  # (white_mask, color_mask) per patch
        self.tracking_ref_hists = None  # colour histogram per patch
        self.last_target_roi = None     # for visualization
        self.last_control_rois = []
        self.occlusion_patch_size = (48, 48)  # finer patch for the alignment step
        self.occlusion_pixel_diff_threshold = 60  # sum of |diff| over the 3 channels
        self.occlusion_roi_cells = 1.4  # ROI side in cells (tight enough for signal)
        self.occlusion_min_zone_fraction = 0.10  # ignore white/color zone if scarcer
        self.occlusion_max_align_shift = 10  # max px shift compensated (warp jitter)
        # Colour-composition gate: a real hand brings a new colour into the ROI, a mere
        # shift does not. Occlusion is only confirmed if the colour histogram also
        # departs from the reference by at least this (Bhattacharyya distance).
        self.occlusion_hist_threshold = 0.30
        
        
    def step(self, undistorted_image, corners, ids):
        self.board_view = self.computeApplyHomography(undistorted_image, corners, ids)
        self.board_contour = self.detectContour(self.board_view)
        # Raw detection state (before inertia) so trial end can be backdated to the
        # last frame in which the board was actually visible
        self.contour_detected_raw = self.board_contour is not None

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
        self.grid_from_reference = False

        # Accumulate the detected border rectangle to build a stable session grid
        if self.board_contour is not None and len(self.board_contour) != 0 and self.contour_detected_raw:
            self.reference_rect_history.append(cv.boundingRect(self.board_contour))
            self.reference_board_rect = tuple(np.median(np.array(self.reference_rect_history), axis=0))

        # The grid is taken from the MEDIAN border rectangle, not the per-frame one.
        # In the warped view the homography already aligns the board every frame, so
        # the detected contour only jitters by detection noise; the median is the true
        # stable position with no lag. This keeps the grid steady (no flicker) AND
        # keeps the touch-detector ROI aligned with the board content (a per-frame grid
        # made the ROI drift over the board and faked occlusions). The per-frame
        # contour is only used at the very start, before enough history is gathered.
        if self.reference_board_rect is not None and len(self.reference_rect_history) >= 5 and self.homography is not None:
            x, y, w, h = self.reference_board_rect
            self.cell_matrix, self.cell_width, self.cell_height = self.computeBoardMatrixFromRect(x, y, w, h, self.board_size)
            self.board_data_dict = self.completeBoardConfig(self.cell_matrix, self.cell_width, self.cell_height, self.board_data_dict)
            self.grid_from_reference = self.board_contour is None
        elif self.board_contour is not None and len(self.board_contour) != 0:
            self.cell_matrix, self.cell_width, self.cell_height = self.computeBoardMatrixFromContour(self.board_contour, self.board_size)
            self.board_data_dict = self.completeBoardConfig(self.cell_matrix, self.cell_width, self.cell_height, self.board_data_dict)
        

    def handleVisualization(self, image, board_contour, board_size, cell_matrix, cell_contours, board_data_dict, fixation_coord_list):
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
                    # log(f'[{i},{j}] in {(x+5-w/2,y+12-h/2)}')
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
        
        if self.display_fixation:
            for fixation_coord in fixation_coord_list:
                fixation_coord_tuple = (int(fixation_coord[0][0]), int(fixation_coord[0][1]))
                cv.circle(display_cfg_board_view, fixation_coord_tuple, radius=10, color=(0,0,255), thickness=-1)
                cv.circle(display_detected_board_view, fixation_coord_tuple, radius=10, color=(0,0,255), thickness=-1)
      
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
            fixation_coord_list=self.fixation_coord_list)

    """
        Projects a set of points from the warped (top) board view back to the
        undistorted image view through the inverse homography
    """
    def warpToUndistorted(self, points, inv_homography):
        pts = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
        return cv.perspectiveTransform(pts, inv_homography)

    def getUndistortedVisualization(self, undistorted_image, corners, ids):
        display_cfg_board_view = undistorted_image
        display_detected_board_view = undistorted_image
        if self.board_contour is not None and self.homography is not None:
            inv_homography = np.linalg.inv(self.homography)

            board_contour_extended = interpolate_points(self.board_contour)
            undistorted_board_contour = self.warpToUndistorted(board_contour_extended, inv_homography).astype(np.int32)

            undistorted_fixation_coord_list = []

            undistorted_cell_matrix = self.cell_matrix
            if undistorted_cell_matrix is not None:
                projected = self.warpToUndistorted(self.cell_matrix.reshape(-1, 2), inv_homography)
                undistorted_cell_matrix = projected.reshape(self.board_size[1], self.board_size[0], 2).astype(int)

            undistorted_cell_contours = None
            if self.cell_contours is not None:
                undistorted_cell_contours = []
                for contour in self.cell_contours:
                    contour_extended = interpolate_points(contour)
                    undistorted_cell_contours.append(self.warpToUndistorted(contour_extended, inv_homography).astype(np.int32))

            display_cfg_board_view, display_detected_board_view = self.handleVisualization(undistorted_image, undistorted_board_contour, self.board_size, undistorted_cell_matrix, undistorted_cell_contours, self.board_data_dict, undistorted_fixation_coord_list)

        if self.display_arucos:
            aruco_list, _, extra_aruco = self.aruco_board_handler.processArucos(corners, ids)

            for aruco_data in extra_aruco:
                aruco_id = np.array([[aruco_data['id']]], dtype=np.int32)
                aruco_corners = aruco_data['points_image'].reshape((1, 4, 2))                    
                cv.aruco.drawDetectedMarkers(image=display_cfg_board_view, corners=[aruco_corners], ids=aruco_id, borderColor=(255,255,0))
                cv.aruco.drawDetectedMarkers(image=display_detected_board_view, corners=[aruco_corners], ids=aruco_id, borderColor=(255,255,0))
                
            for aruco_data in aruco_list:
                aruco_id = np.array([[aruco_data['id']]], dtype=np.int32)
                aruco_corners = aruco_data['points_image'].reshape((1, 4, 2))                    
                cv.aruco.drawDetectedMarkers(image=display_cfg_board_view, corners=[aruco_corners], ids=aruco_id, borderColor=(0,0,255))
                cv.aruco.drawDetectedMarkers(image=display_detected_board_view, corners=[aruco_corners], ids=aruco_id, borderColor=(0,0,255))
            

        # If cannot plot any data...
        return display_cfg_board_view, display_detected_board_view

    def computeApplyHomography(self, undistorted_image, corners, ids):
        display_image = undistorted_image

        self.homography, self.warp_width, self.warp_height, rotated = self.aruco_board_handler.getTransform(undistorted_image, corners, ids)
        self.board_data_dict = self.board_data_dict_rotated if rotated else self.board_data_dict_upright

        if self.homography is not None:
            display_image = cv.warpPerspective(undistorted_image, self.homography, (self.warp_width, self.warp_height))

        return display_image

    """
        Gets image coordinates and returns info about where in the board was it located
        (if its in the board at all)
    """
    def getPixelInfo(self, coordinates_list):
        self.fixation_coord_list = []
        coord_info_list = []
        if coordinates_list and self.cell_matrix is not None \
            and self.board_data_dict is not None:
            log_debug(f"[BoardHandler::getPixelInfo] Check {len(coordinates_list)} fixation for this frame.")

            for coordinates in coordinates_list:
                corrected_coord = self.distortion_handler.correctCoordinates(coordinates, self.homography)
                self.fixation_coord_list.append(corrected_coord)
                idx = self.getCellIndex(corrected_coord)

                if idx[0] is not None:
                    color = self.board_data_dict[idx][0]
                    shape = self.board_data_dict[idx][1]
                    slot = self.board_data_dict[idx][2]
                    board_coord = idx

                    log_debug(f"\t\t· Fixation detected in: {self.board_data_dict[idx]} in {board_coord}.")
                    coord_info_list.append((color, shape, slot, board_coord, corrected_coord))
                else:
                    log_debug(f"\t\t· Fixation not detected, coordinates not detected in board: {corrected_coord}.")
                    coord_info_list.append(('not_board', 'not_board', False, [-1,-1], corrected_coord))
                
        return coord_info_list

    def getShapeCellIndex(self, shape, color, piece=True):
        # print(f"Get Cell Index for {shape = }, {color = }, {piece = }")
        if self.cell_matrix is None:
            return None, None
        for i in range(self.cell_matrix.shape[0]):
            for j in range(self.cell_matrix.shape[1]):
                if (str(self.board_data_dict[(j,i)][0]) == str(color) and
                    str(self.board_data_dict[(j,i)][1]) == str(shape) and
                    bool(self.board_data_dict[(j,i)][2]) == bool(piece)):
                    return [i,j]
        return None, None
    
    def getShapeCoord(self, shape, color, piece=True):
        # print(f"Get Cell Coord for {shape = }, {color = }, {piece = }")
        cell_index = self.getShapeCellIndex(shape, color, piece)
        if cell_index[0] is not None:
            coord = self.cell_matrix[cell_index[0]][cell_index[1]]
            return coord
        else:
            return None, None
        

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
    
    def computeBoardMatrixFromContour(self, board_contour, board_size):
        x, y, w, h = cv.boundingRect(board_contour)
        return self.computeBoardMatrixFromRect(x, y, w, h, board_size)

    def computeBoardMatrixFromRect(self, x, y, w, h, board_size):
        # Float cell sizes: integer division accumulated a remainder of up to
        # board_size-1 px at the right/bottom edges, misclassifying gaze samples
        # on the last cells as not_board
        cell_width = w / board_size[0]
        cell_height = h / board_size[1]

        self.board_origin = (x, y)
        self.board_width = w
        self.board_height = h

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
    
    def getPixelBoardNorm(self, pixel_coord):
        x, y = pixel_coord[0]

        if x is None or y is None:
            return np.array([None, None])

        # Normalized against the detected board area itself, so [0,0] is the board
        # top-left corner and [1,1] its bottom-right corner
        norm_x = (x - self.board_origin[0]) / self.board_width
        norm_y = (y - self.board_origin[1]) / self.board_height

        return np.array([norm_x, norm_y])


    def getCellIndex(self, pixel_coord):
        x, y = pixel_coord[0]

        if x is None or y is None:
            return None, None

        relative_x = x - self.board_origin[0]
        relative_y = y - self.board_origin[1]

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
        pixel_distance = 13
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
        # Lower bounds clamp at 0 (max, not min: with min they were always 0 and the
        # mask accepted any dark pixel, e.g. black clothing entering the view)
        s_margins = [max(0, mean_s-4*std_s), min(255, mean_s+4*std_s)]
        v_margins = [max(0, mean_v-4*std_v), min(255, mean_v+4*std_v)]

        # mean_h/std_h come from the HSV_FULL channel (0-255) but getMaskHue expects
        # degrees (0-360), convert before passing
        h_ref_deg = mean_h * 360.0 / 255.0
        h_eps_deg = 4 * std_h * 360.0 / 255.0
        res = getMaskHue(hue, sat, intensity, h_ref=h_ref_deg, h_epsilon=h_eps_deg, s_margins=s_margins, v_margins = v_margins)
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

    """
        A cell-sized patch with (almost) no saturated nor dark pixels means a blank
        white surface (the sample panel) is covering the board at that point: every
        real cell shows either a colored piece or a colored slot outline
    """
    def isRegionBlank(self, corrected_coord, saturation_threshold=60, dark_threshold=110, blank_fraction=0.97):
        if self.board_view is None or self.cell_width is None:
            return False
        x, y = int(corrected_coord[0][0]), int(corrected_coord[0][1])
        half_w, half_h = int(self.cell_width/2), int(self.cell_height/2)
        x0, x1 = max(0, x-half_w), min(self.board_view.shape[1], x+half_w)
        y0, y1 = max(0, y-half_h), min(self.board_view.shape[0], y+half_h)
        if x1 <= x0 or y1 <= y0:
            return True
        patch = cv.cvtColor(self.board_view[y0:y1, x0:x1], cv.COLOR_BGR2HSV_FULL)
        non_blank = np.logical_or(patch[:,:,1] > saturation_threshold, patch[:,:,2] < dark_threshold)
        return (1.0 - np.count_nonzero(non_blank)/non_blank.size) >= blank_fraction

    ## TARGET AREA OCCLUSION TRACKING (trial end detection)

    """
        True when the target cell is NOT covered by a blank white surface, i.e. the
        sample panel cardboard is no longer sweeping over it. Used to delay the start
        of the touch tracking until the scene has settled after the panel removal.
    """
    def isTargetAreaClear(self, target_cell):
        if self.cell_matrix is None or target_cell is None or target_cell[0] is None:
            return False
        row, col = int(target_cell[0]), int(target_cell[1])
        if not (0 <= row < self.cell_matrix.shape[0] and 0 <= col < self.cell_matrix.shape[1]):
            return False
        cx, cy = self.cell_matrix[row][col]
        return not self.isRegionBlank([[cx, cy]])

    def _clampRoi(self, cx, cy, w, h):
        if self.board_view is None:
            return None
        x0, y0 = int(max(0, cx - w/2)), int(max(0, cy - h/2))
        x1 = int(min(self.board_view.shape[1], cx + w/2))
        y1 = int(min(self.board_view.shape[0], cy + h/2))
        if x1 <= x0 or y1 <= y0:
            return None
        return (x0, y0, x1, y1)

    def _extractPatch(self, roi):
        x0, y0, x1, y1 = roi
        patch = cv.resize(self.board_view[y0:y1, x0:x1], self.occlusion_patch_size, interpolation=cv.INTER_AREA)
        return cv.GaussianBlur(patch, (3, 3), 0).astype(np.float32)

    """
        Starts tracking the appearance of the target cell neighbourhood (target cell
        expanded half a cell on each side). Control cells are spread positions far
        from the target whose median change estimates the global variation. The
        reference patches are captured on the first getTargetOcclusionMeasure call.
    """
    def initTargetTracking(self, target_cell):
        self.clearTargetTracking()
        if target_cell is None or target_cell[0] is None or self.cell_matrix is None:
            return
        self.target_cell = (int(target_cell[0]), int(target_cell[1]))
        rows, cols = self.board_size[1], self.board_size[0]
        candidates = [(0, 0), (0, cols-1), (rows-1, 0), (rows-1, cols-1), (rows//2, cols//2)]
        candidates.sort(key=lambda c: -(abs(c[0]-self.target_cell[0]) + abs(c[1]-self.target_cell[1])))
        self.control_cells = candidates[:3]

    def clearTargetTracking(self):
        self.target_cell = None
        self.control_cells = []
        self.tracking_ref_patches = None
        self.tracking_ref_masks = None
        self.tracking_ref_hists = None
        self.last_target_roi = None
        self.last_control_rois = []

    def _cellRoi(self, cell):
        row, col = cell
        cx, cy = self.cell_matrix[row][col]
        return self._clampRoi(cx, cy, self.cell_width*self.occlusion_roi_cells,
                              self.cell_height*self.occlusion_roi_cells)

    """
        Splits a reference patch into its white (gutters/borders/blank) and colored
        pixels. Thresholds calibrated on the real warped board view, not the print
        render. Used only on the clean reference, so relief/lighting do not affect it.
    """
    def _segmentPatch(self, patch_bgr):
        hsv = cv.cvtColor(np.clip(patch_bgr, 0, 255).astype(np.uint8), cv.COLOR_BGR2HSV_FULL)
        sat, val = hsv[:, :, 1], hsv[:, :, 2]
        white_mask = (sat < 60) & (val > 140)
        color_mask = sat > 90
        return white_mask, color_mask

    """
        Hue-Saturation colour histogram of a patch, invariant to small spatial shifts.
        A mere warp shift keeps the colour composition; a hand brings a new colour in.
    """
    def _patchHist(self, patch_bgr):
        hsv = cv.cvtColor(np.clip(patch_bgr, 0, 255).astype(np.uint8), cv.COLOR_BGR2HSV)
        hist = cv.calcHist([hsv], [0, 1], None, [16, 16], [0, 180, 0, 256])
        cv.normalize(hist, hist, 0, 1, cv.NORM_MINMAX)
        return hist

    """
        Fraction of changed pixels in the target area and the median of the control
        areas, with respect to their reference appearance. Returns None when not
        observable this frame (no valid warp/grid). The ROIs are re-anchored to the
        current grid every frame: warp shifts move grid and content together, so the
        patches stay aligned with the same physical board area.
    """
    def getTargetOcclusionMeasure(self):
        if self.target_cell is None or self.board_view is None \
           or self.homography is None or self.cell_matrix is None:
            return None

        cells = [self.target_cell] + self.control_cells
        rois = [self._cellRoi(cell) for cell in cells]
        if any(roi is None for roi in rois):
            return None
        self.last_target_roi = rois[0]
        self.last_control_rois = rois[1:]

        patches = [self._extractPatch(roi) for roi in rois]
        if self.tracking_ref_patches is None:
            self.tracking_ref_patches = patches
            self.tracking_ref_masks = [self._segmentPatch(patch) for patch in patches]
            self.tracking_ref_hists = [self._patchHist(patch) for patch in patches]
            return (0.0, 0.0)

        def changedFraction(patch, reference, masks, ref_hist):
            # GATE 1 - colour composition: a real hand brings a new colour into the
            # ROI; a mere warp shift keeps the same colours. If the colour histogram
            # barely changed, it is a shift, not an occlusion -> 0.
            if cv.compareHist(ref_hist, self._patchHist(patch), cv.HISTCMP_BHATTACHARYYA) < self.occlusion_hist_threshold:
                return 0.0
            # GATE 2 - compensate a small whole-ROI shift before measuring change. The
            # warp jitters a few px between frames (the arucos are detected with slight
            # variation), so the board content "dances" under the fixed ROI; aligning
            # the patch to its reference first means a residual shift leaves little
            # change, while a real hand over the piece cannot be aligned away.
            ref_gray = cv.cvtColor(np.clip(reference, 0, 255).astype(np.uint8), cv.COLOR_BGR2GRAY).astype(np.float32)
            pat_gray = cv.cvtColor(np.clip(patch, 0, 255).astype(np.uint8), cv.COLOR_BGR2GRAY).astype(np.float32)
            try:
                (dx, dy), _ = cv.phaseCorrelate(ref_gray, pat_gray)
            except cv.error:
                dx, dy = 0.0, 0.0
            s = self.occlusion_max_align_shift
            dx, dy = max(-s, min(s, dx)), max(-s, min(s, dy))
            if abs(dx) > 0.3 or abs(dy) > 0.3:
                M = np.float32([[1, 0, -dx], [0, 1, -dy]])
                patch = cv.warpAffine(patch, M, (patch.shape[1], patch.shape[0]), borderMode=cv.BORDER_REFLECT)
            diff = np.abs(patch - reference).sum(axis=2)
            changed = diff > self.occlusion_pixel_diff_threshold
            # Change measured separately on the white (gutters) and the colored (piece)
            # pixels of the cell, then averaged. A finger reaching the target covers the
            # colored piece reliably but the white gutters only sometimes; requiring BOTH
            # (min) missed those touches, so the MEAN of the two zones is used. Shadows
            # (which would change both zones) are already rejected upstream by the colour
            # histogram gate and by the target-vs-control separation. Fall back to the
            # whole patch if a zone is too scarce to be reliable (e.g. a piece with almost
            # no white border).
            min_px = self.occlusion_min_zone_fraction * changed.size
            zone_fractions = [float(changed[mask].mean()) for mask in masks if mask.sum() >= min_px]
            if not zone_fractions:
                return float(changed.mean())
            return float(np.mean(zone_fractions))

        fractions = [changedFraction(patch, ref, masks, ref_hist)
                     for patch, ref, masks, ref_hist in zip(patches, self.tracking_ref_patches, self.tracking_ref_masks, self.tracking_ref_hists)]
        frac_target = fractions[0]
        frac_control = float(np.median(fractions[1:]))

        # Slow reference update while the scene is calm, to absorb lighting drift.
        # Masks are not recomputed: the white/color structure of the cell is stable.
        if frac_target < 0.05 and frac_control < 0.05:
            self.tracking_ref_patches = [0.95*ref + 0.05*patch
                                         for ref, patch in zip(self.tracking_ref_patches, patches)]

        return (frac_target, frac_control)

    def detectSlots(self, image):
        pass
        ## Detection of squares and slots :)
        # might be unnecesary
        # detected_board_data = []
        # contours_dict = detectColorSquares(capture, color_dict=color_dict, colors_list=colors_list, display_image=None)
        # for color, contour_list in contours_dict.items():
        #     for contour in contour_list:
        #         # log(f'Detect shape for {color} and contour {contour}')
        #         is_slot, center = isSlot(capture, contour, color_dict[color], colors_list[color], display_image=None)
        #         detected_board_data.append([is_slot, center])
