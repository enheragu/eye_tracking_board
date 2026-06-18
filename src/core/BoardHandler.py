#!/usr/bin/env python3
# encoding: utf-8

import math
from collections import deque

import cv2 as cv
import numpy as np
from skimage.metrics import structural_similarity as _ssim

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
        self.border_thr_ema = None           # temporally smoothed black-border threshold (EMA)

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
        self.touch_warm_target = False
        self.control_cells = []
        self.tracking_ref_patches = None
        self.tracking_ref_masks = None  # (white_mask, color_mask) per patch
        self.tracking_ref_hists = None  # colour histogram per patch
        self.last_target_roi = None     # for visualization
        self.last_control_rois = []

        ## Session-level CLEAN board template (v1.2.0). The board pieces are static, so a
        # clean warped view captured between/within trials (board fully visible, no panel,
        # no hand) is a valid touch reference for ALL trials -- crucially for EDGE targets
        # where the hand covers the target from the very start of the trial and there is no
        # clean per-trial window to capture a reference. Refreshed (blended) whenever the
        # board is verified clean, to track lighting drift. PERSISTS across trials (not
        # cleared in clearTargetTracking).
        self.session_template = None
        self._template_blend_every = 15   # blend the template every N clean frames only
        self._template_blend_counter = 0  # (lighting drifts slowly; a full-frame float
                                          #  blend every frame was needless cost)

        ## Whole-board occlusion baseline (hand_exit detection, v1.2.0). Unlike the
        # binary border contour -- which reappears MID-reach because the hand is in the
        # centre and the white frame is visible again (measured ambiguity, see doc) --
        # a coarse change measure over the WHOLE board area stays high while any large
        # object (the hand/arm) covers the board, and only returns to baseline when the
        # hand actually leaves. Its sustained return marks hand_exit independently of
        # the touch (which is what coupled hand_exit to the touch in v1.1.0). Cheap:
        # a single 64x36 grayscale diff per frame.
        self.board_occ_ref = None
        self.board_occ_size = (64, 36)
        self.board_occ_diff_threshold = 35  # per-pixel grayscale change to count
        # Last whole-board change mask + its crop, kept so the per-cell occlusion map
        # (wrong-piece detection) can be derived from it for FREE (no per-cell patch/SSIM).
        self._board_occ_changed = None
        self._board_occ_crop = None
        # Debug-only capture of the target-cell touch masks (patch/ref/diff/edge/ssim/changed)
        # for the documentation figure. Off by default -> zero cost in normal runs.
        self._dbg_masks_on = False
        self._dbg_masks = None
        # Debug-only capture of the WHOLE-BOARD occlusion masks (patch/ref/diff/changed) -- the
        # board_occ equivalent of the target masks, for the documentation figure. Off by default.
        self._dbg_board_masks_on = False
        self._dbg_board_masks = None

        ## Sample-panel mask in warped-board coordinates (v1.2.0). The panel is a KNOWN
        # occluder while it is being removed (not the hand), so it is excluded from the
        # border (contour) detection -- letting the border be found from the visible part
        # as the panel retracts -- and from the hand-occlusion measure (so it neither
        # fakes hand entry nor contaminates the reference). Projected from the panel
        # polygon through the board homography each frame; None when no panel is detected.
        self.panel_mask_warp = None
        self.occlusion_patch_size = (48, 48)  # finer patch for the alignment step
        self.occlusion_pixel_diff_threshold = 60  # sum of |diff| over the 3 channels
        self.occlusion_edge_threshold = 50        # gradient-magnitude rise = hand texture/edges (warm targets)
        self.occlusion_ssim_threshold = 0.55      # per-pixel SSIM below this = structure broken (a hand)
        self.occlusion_roi_cells = 1.4  # ROI side in cells (tight enough for signal)
        self.occlusion_min_zone_fraction = 0.10  # ignore white/color zone if scarcer
        self.occlusion_max_align_shift = 10  # max px shift compensated (warp jitter)
        # Colour-composition gate: a real hand brings a new colour into the ROI, a mere
        # shift does not. Occlusion is only confirmed if the colour histogram also
        # departs from the reference by at least this (Bhattacharyya distance).
        self.occlusion_hist_threshold = 0.30
        
        
    def step(self, undistorted_image, corners, ids, panel_polygon=None):
        self.board_view = self.computeApplyHomography(undistorted_image, corners, ids)
        self.panel_mask_warp = self._panelMaskWarp(panel_polygon)
        self.board_contour = self.detectContour(self.board_view, ignore_mask=self.panel_mask_warp)
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

        self.refreshSessionTemplate()

    ## Maintains the session-level clean board template: captures/blends the current
    # warped view whenever the board is clearly clean (full contour detected this frame
    # and no panel overlapping it). The slow blend tracks lighting drift; it never
    # captures while the hand crosses the border (contour lost) or the panel is present.
    def refreshSessionTemplate(self):
        if self.board_view is None or not self.contour_detected_raw or self.panel_mask_warp is not None:
            return
        if self.session_template is None or self.session_template.shape != self.board_view.shape:
            self.session_template = self.board_view.astype(np.float32)
            self._template_blend_counter = 0
            return
        # Blend only every N clean frames: lighting drift is slow, and a full-frame float
        # blend on every frame was the main per-frame cost added in v1.2 (measured).
        self._template_blend_counter += 1
        if self._template_blend_counter >= self._template_blend_every:
            self._template_blend_counter = 0
            self.session_template = 0.9 * self.session_template + 0.1 * self.board_view.astype(np.float32)
        

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
        # NOTE: returns [row, col] (i, j) -- the OPPOSITE order to getCellIndex, which returns
        # (col, row). board_data_dict is keyed by (col, row). Callers must convert accordingly
        # (e.g. store_results builds target_colrow = [target_cord[1], target_cord[0]]).
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

    ## Projects the sample-panel polygon (undistorted-image coordinates, from
    # PanelHandler.getPanelPolygon) into the warped board view, as a filled mask. Used
    # to exclude the panel (a known occluder) from the border detection and the hand
    # occlusion. Returns None when there is no panel or no homography this frame.
    def _panelMaskWarp(self, panel_polygon):
        if panel_polygon is None or self.homography is None or self.board_view is None:
            return None
        pts = cv.perspectiveTransform(np.asarray(panel_polygon, np.float32).reshape(-1, 1, 2),
                                      self.homography).reshape(-1, 2)
        mask = np.zeros(self.board_view.shape[:2], np.uint8)
        cv.fillPoly(mask, [pts.astype(np.int32)], 255)
        return mask if mask.any() else None

    ## FUNCTIONS BASED ON DETECTION OVER IMAGE
    def detectContour(self, image, ignore_mask=None):
        hsv_image = cv.cvtColor(image, cv.COLOR_BGR2HSV_FULL)
        hue, sat, intensity = cv.split(hsv_image)

        # Black-border threshold sampled from the live board margins (adaptive to lighting,
        # per frame -- no stale/circular dependency on a clean template).
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

        # ROBUST center (median) + TEMPORALLY STABLE spread/threshold. Two failure modes are
        # addressed: (1) spatially, a hand entering through a margin is a MINORITY of the
        # 4-edge perimeter -> the median ignores it (mean drifted and flooded the mask, 008
        # f2184); (2) temporally, the border is the SAME frame to frame, so recomputing the
        # threshold fresh each frame jittered the mask -> an EMA across frames stabilises it.
        # Width uses 6*MAD (~4*std for a normal) so it is NOT tighter than the old 4*std
        # (4*MAD was ~2.7*std -> too strict, more flicker, fewer starts -- measured).
        med = np.median(reference_edges, axis=0)
        mad = np.median(np.abs(reference_edges - med), axis=0) * 1.4826  # ~std for a normal
        # threshold bounds this frame: [h_ref_deg, h_eps_deg, s_lo, s_hi, v_lo, v_hi]
        thr_now = np.array([
            med[0] * 360.0 / 255.0, 6*mad[0] * 360.0 / 255.0,
            max(0, med[1]-6*mad[1]), min(255, med[1]+6*mad[1]),
            max(0, med[2]-6*mad[2]), min(255, med[2]+6*mad[2])])
        if self.border_thr_ema is None:
            self.border_thr_ema = thr_now
        else:
            self.border_thr_ema = 0.85 * self.border_thr_ema + 0.15 * thr_now
        h_ref_deg, h_eps_deg, s_lo, s_hi, v_lo, v_hi = self.border_thr_ema
        s_margins = [s_lo, s_hi]
        v_margins = [v_lo, v_hi]
        res = getMaskHue(hue, sat, intensity, h_ref=h_ref_deg, h_epsilon=h_eps_deg, s_margins=s_margins, v_margins = v_margins)
        # Exclude the sample panel (known occluder while it is being removed): its white
        # back would otherwise be taken as board border and break the rectangle, or
        # delay the border detection until it has fully cleared the edge.
        if ignore_mask is not None:
            res[ignore_mask > 0] = 0
        # cv.imshow(f'border_mask', res)

        # PRIMARY board contour: external rectangle taken DIRECTLY on the colour mask
        # (no Canny). Canny double-edged the thick band and fragmented the contour
        # (measured: 12/81 detections vs 43/81 without, on a peeling-piece border). NO
        # morphological close here: a close bridges the narrow gap a hand opens and would
        # blind the cut (motor onset = loss of this contour); the close is only for the
        # separate start-gate signal below.
        contours, hierarchy = cv.findContours(res, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        board_contour = None
        for contour in contours:
            shape, approx = checkShape(contour, area_filter=[700000, math.inf])
            if shape == 'rectangle':
                board_contour = approx
                area = cv.contourArea(contour)
                break

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

    def _extractPatch(self, roi, source=None):
        if source is None:
            source = self.board_view
        x0, y0, x1, y1 = roi
        patch = cv.resize(source[y0:y1, x0:x1], self.occlusion_patch_size, interpolation=cv.INTER_AREA)
        return cv.GaussianBlur(patch, (3, 3), 0).astype(np.float32)

    """
        Starts tracking the appearance of the target cell neighbourhood (target cell
        expanded half a cell on each side). Control cells are spread positions far
        from the target whose median change estimates the global variation. The
        reference patches are captured on the first getTargetOcclusionMeasure call.
    """
    def initTargetTracking(self, target_cell, target_color=None):
        self.clearTargetTracking()
        if target_cell is None or target_cell[0] is None or self.cell_matrix is None:
            return
        # Per-target detection profile (v1.2): for WARM targets (red/yellow ~ skin hue)
        # the colour-composition gate (GATE 1) cannot tell a warm-toned hand from the
        # warm piece, so it wrongly suppresses the touch (measured: row4 red/yellow ~62%
        # vs blue 87%). For those we de-weight GATE 1 and rely on the brightness/texture
        # change (GATE 2) + the control separation. Cool targets keep GATE 1 (it works).
        self.touch_warm_target = str(target_color) in ('red', 'yellow')
        tr, tc = int(target_cell[0]), int(target_cell[1])      # (row, col)
        self.target_cell = (tr, tc)
        rows, cols = self.board_size[1], self.board_size[0]
        bd = self.board_data_dict
        def color_of(r, c):                                    # board_data_dict keyed (col,row)
            v = bd.get((c, r)) if bd else None
            return str(v[0]) if v else None
        tgt_color = str(target_color) if target_color is not None else color_of(tr, tc)
        def has_piece(r, c):                                   # 3rd field True = piece, False = empty slot
            v = bd.get((c, r)) if bd else None
            return bool(v[2]) if (v and len(v) > 2) else False
        # The arm reaches from the NEAR edge (bottom = max row index) toward the target, so cells
        # NEARER the participant than the target and ~in its column lie in the forearm's path: the
        # arm inflates their change and falsely cancels the (low-contrast, warm) target separation
        # (`control_ge`). Controls are scored to be: (1) SAME colour as the target, so the occlusion
        # sensitivity matches (a warm hand changes a warm control as little as the warm target,
        # instead of a high-contrast control over-registering it); (2) cells WITH A PIECE, never an
        # empty slot (a slot's white inner area over-registers any hand); (3) OUTSIDE the arm
        # corridor; (4) far from the target and spread, for a fair global-variation estimate. Falls
        # back to the geometric farthest cells (legacy) if too few same-colour cells qualify.
        def in_corridor(r, c):
            return r >= tr and abs(c - tc) <= 1
        same = [(r, c) for r in range(rows) for c in range(cols)
                if (r, c) != (tr, tc) and color_of(r, c) == tgt_color
                and (abs(r - tr) + abs(c - tc)) >= 2]
        if len(same) >= 3:
            def score(cell):
                r, c = cell
                return ((2 if has_piece(r, c) else 0)          # piece >> slot (slot = white over-registers)
                        + (1 if not in_corridor(r, c) else 0), # outside the arm corridor
                        abs(r - tr) + abs(c - tc))             # then far from the target
            pool = sorted(same, key=score, reverse=True)
        else:
            legacy = [(0, 0), (0, cols-1), (rows-1, 0), (rows-1, cols-1), (rows//2, cols//2)]
            legacy.sort(key=lambda c: -(abs(c[0]-tr) + abs(c[1]-tc)))
            pool = same + [c for c in legacy if c not in same]
        chosen = []
        for cell in pool:                                      # keep the priority order, add spread
            if len(chosen) >= 3:
                break
            if all(abs(cell[0]-ch[0]) + abs(cell[1]-ch[1]) >= 2 for ch in chosen):
                chosen.append(cell)
        for cell in pool:                                      # top up if the spread filter left < 3
            if len(chosen) >= 3:
                break
            if cell not in chosen:
                chosen.append(cell)
        self.control_cells = chosen[:3]

    def clearTargetTracking(self):
        self.target_cell = None
        self.touch_warm_target = False
        self.control_cells = []
        self.tracking_ref_patches = None
        self.tracking_ref_masks = None
        self.tracking_ref_hists = None
        self.last_target_roi = None
        self.last_control_rois = []
        self.board_occ_ref = None

    """
        Coarse fraction of the board area that changed vs a clean reference captured
        on the first call (trial start, board clear). The hand/arm over the board
        keeps this high through the whole reach -- including mid-reach, when the border
        contour misleadingly reappears -- so its sustained return to baseline is a
        robust hand_exit signal that does NOT depend on the touch. Returns None when
        the board pose/grid is not available this frame (few ArUcos): the caller then
        falls back to the contour-based logic. A 64x36 grayscale diff: negligible cost.
    """
    def getBoardOcclusionMeasure(self):
        if self.board_view is None or self.homography is None \
           or self.board_origin is None or self.board_width is None or self.board_height is None:
            return None
        x, y = int(self.board_origin[0]), int(self.board_origin[1])
        x0, y0 = max(0, x), max(0, y)
        x1 = min(self.board_view.shape[1], x + int(self.board_width))
        y1 = min(self.board_view.shape[0], y + int(self.board_height))
        if x1 - x0 < 8 or y1 - y0 < 8:
            return None
        small = cv.resize(self.board_view[y0:y1, x0:x1], self.board_occ_size, interpolation=cv.INTER_AREA)
        gray = cv.GaussianBlur(cv.cvtColor(small, cv.COLOR_BGR2GRAY), (3, 3), 0).astype(np.float32)
        if self.board_occ_ref is None:
            self.board_occ_ref = gray
            return 0.0
        changed = np.abs(gray - self.board_occ_ref) > self.board_occ_diff_threshold
        # Exclude the sample panel (known occluder): it must not count as hand occlusion.
        if self.panel_mask_warp is not None:
            pm = cv.resize(self.panel_mask_warp[y0:y1, x0:x1], self.board_occ_size,
                           interpolation=cv.INTER_NEAREST) > 0
            changed = changed & ~pm        # the panel never counts as occlusion (also per-cell)
            valid = ~pm
            frac = float(changed[valid].mean()) if valid.any() else 0.0
        else:
            frac = float(changed.mean())
        # Keep the mask + crop so getCellOcclusionMap can reuse it (no extra work this frame).
        self._board_occ_changed = changed
        self._board_occ_crop = (x0, y0, x1 - x0, y1 - y0)
        # Debug: keep the whole-board intermediate masks (current board, clean reference, abs diff,
        # final change mask) so the documentation figure can show what the board_occ measure sees.
        if self._dbg_board_masks_on:
            self._dbg_board_masks = {'patch': np.clip(small, 0, 255).astype(np.uint8),
                                     'ref': np.clip(self.board_occ_ref, 0, 255).astype(np.uint8),
                                     'diff': np.abs(gray - self.board_occ_ref),
                                     'changed': changed.copy()}
        # Absorb slow lighting drift only while the scene is calm (never while occluded).
        # Do NOT let the sample panel leak into the reference: keep the pre-panel value on
        # panel pixels, otherwise the reference absorbs the panel and marks false change when
        # it is removed.
        if frac < 0.03:
            updated = 0.9 * self.board_occ_ref + 0.1 * gray
            if self.panel_mask_warp is not None:
                updated[pm] = self.board_occ_ref[pm]
            self.board_occ_ref = updated
        return frac

    def getCellOcclusionMap(self):
        """Per-cell hand occlusion, REUSING the whole-board change mask (board_occ): for each
        board cell, the fraction of changed pixels in its region of the 64x36 mask. Cheap (no
        per-cell patch / phaseCorrelate / SSIM -- that gauntlet is only for confirming the
        TARGET touch). Lets us find which piece the hand is actually over when the target is not
        the one being touched (wrong-piece detection). Returns a (rows x cols) array, or None."""
        changed, crop = self._board_occ_changed, self._board_occ_crop
        if changed is None or crop is None or self.cell_matrix is None:
            return None
        gh, gw = changed.shape
        x0, y0, bw, bh = crop
        if bw <= 0 or bh <= 0:
            return None
        rows, cols = self.cell_matrix.shape[0], self.cell_matrix.shape[1]
        hw = max(1, int(self.cell_width / bw * gw / 2.0))
        hh = max(1, int(self.cell_height / bh * gh / 2.0))
        occ = np.zeros((rows, cols), np.float32)
        for i in range(rows):
            for j in range(cols):
                cx, cy = self.cell_matrix[i][j]
                sx = int((cx - x0) / bw * gw)
                sy = int((cy - y0) / bh * gh)
                patch = changed[max(0, sy - hh):sy + hh + 1, max(0, sx - hw):sx + hw + 1]
                occ[i, j] = float(patch.mean()) if patch.size else 0.0
        return occ

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
            # Prefer the LIVE per-trial reference when THIS frame is clean (contour
            # detected: the board is visible and the hand is not over it) -- it tracks the
            # exact current lighting and is the behaviour that already worked on most
            # trials. Fall back to the SESSION TEMPLATE only when there is no clean window
            # (edge targets: the hand covers the target from the start, so the contour is
            # already lost at activation) -- that is exactly where the template rescues the
            # touch, without disturbing the trials that had a clean window of their own.
            use_template = (not self.contour_detected_raw) and self.session_template is not None
            if use_template:
                ref_src = np.clip(self.session_template, 0, 255).astype(np.uint8)
                self.tracking_ref_patches = [self._extractPatch(roi, ref_src) for roi in rois]
            else:
                self.tracking_ref_patches = patches
            self.tracking_ref_masks = [self._segmentPatch(patch) for patch in self.tracking_ref_patches]
            self.tracking_ref_hists = [self._patchHist(patch) for patch in self.tracking_ref_patches]
            if not use_template:
                return (0.0, 0.0)
            # else: fall through and measure the live patches vs the template reference

        def changedFraction(patch, reference, masks, ref_hist, is_target=False):
            # GATE 1 - colour composition: a real hand brings a new colour into the
            # ROI; a mere warp shift keeps the same colours. If the colour histogram
            # barely changed, it is a shift, not an occlusion -> 0. SKIPPED for warm
            # targets (red/yellow ~ skin hue): there a warm-toned hand barely changes the
            # hue histogram, so GATE 1 would wrongly zero a real touch; GATE 2 (brightness
            # /texture) + the control separation carry the detection instead.
            if not self.touch_warm_target:
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
            # EDGE/TEXTURE variation (v1.2): a hand adds structure (finger edges, shadows,
            # nail) that a flat painted piece lacks -- a COLOUR-INDEPENDENT cue that catches
            # a warm hand over a warm piece, where the colour/brightness change is tiny, and
            # boosts the weak touches of any colour. Applied to the TARGET only (never the
            # control cells: the arm crossing them adds edges too and would shrink the
            # target-vs-control separation). For COOL targets GATE 1 already gated above, so
            # the edge only adds when a colour change is present -> no false positives.
            if is_target:
                pat_gray_a = cv.cvtColor(np.clip(patch, 0, 255).astype(np.uint8), cv.COLOR_BGR2GRAY).astype(np.float32)
                pgrad = cv.magnitude(cv.Sobel(pat_gray_a, cv.CV_32F, 1, 0, ksize=3), cv.Sobel(pat_gray_a, cv.CV_32F, 0, 1, ksize=3))
                rgrad = cv.magnitude(cv.Sobel(ref_gray, cv.CV_32F, 1, 0, ksize=3), cv.Sobel(ref_gray, cv.CV_32F, 0, 1, ksize=3))
                edge = pgrad - rgrad
                changed = changed | (edge > self.occlusion_edge_threshold)
                # SSIM structural change: a hand breaks the piece's structure -> low SSIM,
                # robust to noise/lighting and INDEPENDENT of colour (catches warm hands on
                # warm pieces, where pixel/edge change is small). Low-SSIM pixels = changed.
                smap = None
                try:
                    _, smap = _ssim(ref_gray.astype(np.uint8), pat_gray_a.astype(np.uint8),
                                    full=True, data_range=255)
                    changed = changed | (smap < self.occlusion_ssim_threshold)
                except Exception:
                    pass
                # Debug: keep the intermediate target-cell masks (gated, no cost in normal runs)
                # so a documentation figure can show what the touch detector actually sees.
                if self._dbg_masks_on:
                    self._dbg_masks = {'patch': np.clip(patch, 0, 255).astype(np.uint8),
                                       'ref': np.clip(reference, 0, 255).astype(np.uint8),
                                       'diff': diff, 'edge': edge, 'ssim': smap,
                                       'changed': changed.copy()}
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

        fractions = [changedFraction(patch, ref, masks, ref_hist, is_target=(i == 0))
                     for i, (patch, ref, masks, ref_hist) in enumerate(zip(patches, self.tracking_ref_patches, self.tracking_ref_masks, self.tracking_ref_hists))]
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
