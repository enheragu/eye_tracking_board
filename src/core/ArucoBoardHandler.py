
#!/usr/bin/env python3
# encoding: utf-8

import math

import cv2 as cv
import numpy as np
from scipy.spatial import distance as dist

import yaml
from yaml.loader import SafeLoader

from src.core.utils import projectCenter
from src.core.utils import log

"""
    Gray-world chromatic correction in LAB space, weighted by luminance.
    Stats are computed over a subsampled image (identical result, 16x cheaper).
"""
def ARUCOColorCorrection(input_img):
    result = cv.cvtColor(input_img, cv.COLOR_BGR2LAB)
    subsampled = result[::4, ::4]
    avg_a = np.average(subsampled[:, :, 1])
    avg_b = np.average(subsampled[:, :, 2])
    luma_weight = (result[:, :, 0].astype(np.float32) / 255.0) * 1.1
    a_corrected = result[:, :, 1].astype(np.float32) - (avg_a - 128) * luma_weight
    b_corrected = result[:, :, 2].astype(np.float32) - (avg_b - 128) * luma_weight
    result[:, :, 1] = np.clip(a_corrected, 0, 255).astype(np.uint8)
    result[:, :, 2] = np.clip(b_corrected, 0, 255).astype(np.uint8)
    return cv.cvtColor(result, cv.COLOR_LAB2BGR)


def rotate_points_180_around_center(points, board_center):
    rotation_matrix = np.array([
        [-1, 0],
        [0, -1]
    ])

    translated_points = points - board_center
    rotated_points = np.dot(translated_points, rotation_matrix.T) + board_center

    return rotated_points

## Detector configuration built once (it was being rebuilt on every call)
ARUCO_DICTIONARY = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_ARUCO_ORIGINAL)
ARUCO_PARAMS = cv.aruco.DetectorParameters()
ARUCO_PARAMS.adaptiveThreshWinSizeMin = 5
ARUCO_PARAMS.adaptiveThreshWinSizeMax = 15
ARUCO_PARAMS.adaptiveThreshConstant = 7
ARUCO_PARAMS.minMarkerPerimeterRate = 0.05
ARUCO_PARAMS.maxMarkerPerimeterRate = 4.0
# Subpixel refinement is already performed by the detector itself; the manual
# cornerSubPix pass that was applied afterwards was redundant
ARUCO_PARAMS.cornerRefinementMethod = cv.aruco.CORNER_REFINE_SUBPIX

def detectAllArucos(undistorted_frame, aruco_dictionary = ARUCO_DICTIONARY):
    gray_image = cv.cvtColor(undistorted_frame, cv.COLOR_BGR2GRAY)  # transforms to gray level
    corners, ids, rejectedImgPoints = cv.aruco.detectMarkers(gray_image, aruco_dictionary, parameters=ARUCO_PARAMS)
    return corners, ids

class ArucoBoardHandler:
    def __init__(self, arucoboard_cfg_path, colors_list, color, cameraMatrix = None, distCoeffs = None, shape = '', estimate_rotation = True, min_markers = None):

        self.colors_list = colors_list
        self.color = color
        self.shape = shape
        # Minimum detected configured markers to estimate the board pose. None falls
        # back to ~half of the configured markers; a lower absolute value keeps the
        # pose alive while the hand covers part of the board (helps occlusion-based
        # trial end and early gaze projection)
        self.min_markers = min_markers
        self.arucos_detected = []

        self.cameraMatrix = cameraMatrix
        self.distCoeffs = distCoeffs

        self.aruco_config, self.board_poinst_3d, self.board_poinst_3d_rotated, self.aruco_side = self.parseCFGPanelData(arucoboard_cfg_path)
        self.config_by_id = {int(cfg['id']): cfg for cfg in self.aruco_config}
        self.config_ids = set(self.config_by_id.keys())

        ## 180 degree rotation state. Pose estimation is only needed for the game
        # board (the cell map flips with it); panels skip it (estimate_rotation=False)
        self.estimate_rotation = estimate_rotation
        self.rotated_state = False
        self.rotation_flip_counter = 0
        self.rotation_flip_threshold = 3

    def __repr__(self):
        str_format = f"ArucoBoardHandler ({self.color} - {self.shape}): ["
        for aruco_config_item in self.aruco_config:
            str_format+= f"'id':{aruco_config_item['id']};"
        str_format+= f"]"
        return str_format
    
    def parseCFGPanelData(self, arucoboard_cfg_path):
        aruco_3d_data = {}
        sheet_points_3d = None
        with open(arucoboard_cfg_path) as file:
            data = yaml.load(file, Loader=SafeLoader)
            board_size = data['board_size']
            board_position = data['board_position']
            aruco_side = data['aruco_side']

            board_center = np.array([board_position[0] + board_size[0] / 2, board_position[1] + board_size[1] / 2])        

            aruco_3d_data = []
            for marker_cfg in data['marker_configuration']:
                aruco_3d_data.append({})
                aruco_3d_data[-1] = marker_cfg
                aruco_3d_data[-1]['points_3d'] = np.array([
                    [marker_cfg['position'][0], marker_cfg['position'][1]], #, marker_cfg['position'][2]],                      # Esquina superior izquierda
                    [marker_cfg['position'][0], marker_cfg['position'][1]+aruco_side], #, marker_cfg['position'][2]],           # Esquina inferior izquierda
                    [marker_cfg['position'][0]+aruco_side, marker_cfg['position'][1]], #, marker_cfg['position'][2]],           # Esquina superior derecha
                    [marker_cfg['position'][0]+aruco_side, marker_cfg['position'][1]+aruco_side] #, marker_cfg['position'][2]] # Esquina inferior derecha
                ], dtype=np.float32)

                aruco_3d_data[-1]['points_3d_rotated'] = rotate_points_180_around_center(aruco_3d_data[-1]['points_3d'], board_center)


            board_points_3d = np.array([
                [board_position[0], board_position[1]], #, board_position[2]],                      # Esquina superior izquierda
                [board_position[0], board_position[1]+board_size[1]], #, board_position[2]],           # Esquina inferior izquierda
                [board_position[0]+board_size[0], board_position[1]], #, board_position[2]],           # Esquina superior derecha
                [board_position[0]+board_size[0], board_position[1]+board_size[1]] #, board_position[2]] # Esquina inferior derecha
            ], dtype=np.float32)


            board_points_3d_rotated = rotate_points_180_around_center(board_points_3d, board_center)

        
        return aruco_3d_data, board_points_3d, board_points_3d_rotated, aruco_side


    def processArucos(self, corners, ids):
        detected_arucos_list = []
        self.arucos_detected = []
        extra_aruco_list = []

        if ids is None:
            return [], self.rotated_state, []

        matched_corners = []
        for index, marker_id in enumerate(np.asarray(ids).flatten()):
            marker_id = int(marker_id)
            aruco_data = self.config_by_id.get(marker_id)
            if aruco_data is not None:
                current_data = dict(aruco_data)  # shallow copy, only points_image is added
                current_data['points_image'] = corners[index][0]
                detected_arucos_list.append(current_data)
                self.arucos_detected.append(marker_id)
                matched_corners.append(corners[index])
            else:
                extra_aruco_list.append({'id': marker_id, 'points_image': corners[index][0]})

        need_rotation = self.checkRotation(matched_corners)
        return detected_arucos_list, need_rotation, extra_aruco_list

    """
        Decides if the board is seen rotated 180 degrees. Only this handler's own
        markers vote (foreign markers in the frame used to pollute the count), and
        the decision has hysteresis: physically the board cannot flip mid trial.
    """
    def checkRotation(self, matched_corners):
        if not self.estimate_rotation or not matched_corners:
            return self.rotated_state

        rvecs, tvecs, _ = cv.aruco.estimatePoseSingleMarkers(matched_corners, self.aruco_side, self.cameraMatrix, self.distCoeffs)
        rotated_count = 0
        for rvec in rvecs:
            R, _ = cv.Rodrigues(rvec)  # Rotation matrix
            yaw = np.arctan2(R[1, 0], R[0, 0])
            if abs(yaw) > np.pi / 2:
                rotated_count += 1

        majority_rotated = rotated_count > len(matched_corners) / 2.0
        if majority_rotated != self.rotated_state:
            self.rotation_flip_counter += 1
            if self.rotation_flip_counter >= self.rotation_flip_threshold:
                self.rotated_state = majority_rotated
                self.rotation_flip_counter = 0
        else:
            self.rotation_flip_counter = 0

        return self.rotated_state

    def getTransform(self, undistorted_frame, corners, ids):

        homography = None
        _, (warp_width, warp_height) = rescale_3d_points(self.board_poinst_3d, undistorted_frame.shape)

        # Minimum configured markers to estimate the pose: an explicit min_markers
        # (board) or ~half of them by default (panels)
        aruco_detected_min = self.min_markers if self.min_markers is not None else max(1, len(self.aruco_config)*0.49)

        # Cheap early exit on the detected id set, before any matching/pose work.
        # With 10+ panel configurations checked per frame this skips almost all work
        if ids is None or len(self.config_ids.intersection(np.asarray(ids).flatten().astype(int).tolist())) < aruco_detected_min:
            self.arucos_detected = []
            return None, int(warp_width), int(warp_height), self.rotated_state

        detected_aruco_list, rotated, _ = self.processArucos(corners, ids)

        if len(detected_aruco_list) >= aruco_detected_min:
            aruco_corners_image = np.array([detected_aruco['points_image'] for detected_aruco in detected_aruco_list])

            ## Theres one marker rotated all the time...ups
            # Hack to work with rotated bard...
            aruco_coord_tag = 'points_3d_rotated' if rotated else 'points_3d'
            board_poinst_3d = self.board_poinst_3d_rotated if rotated else self.board_poinst_3d 
            
            aruco_corners_image = []
            aruco_corners_3d = []
            for detected_aruco in detected_aruco_list:
                aruco_corners_image.append(sort_points_clockwise(detected_aruco['points_image']))
                aruco_corners_3d.append(sort_points_clockwise(detected_aruco[aruco_coord_tag]))

            if aruco_corners_image != [] and aruco_corners_3d != []:
                aruco_corners_image = np.array(aruco_corners_image, dtype=np.float32)
                aruco_corners_3d = np.array(aruco_corners_3d, dtype=np.float32)

                homography, warp_width, warp_height = aruco_board_transform(
                                        aruco_image_contours=aruco_corners_image,
                                        aruco_3d_contours=aruco_corners_3d,
                                        board_3d_contours=board_poinst_3d,
                                        img_shape=undistorted_frame.shape)

        return homography, int(warp_width), int(warp_height), rotated


    def handleVisualization(self, image, corners, ids):
        aruco_list, rotated, other_arucos = self.processArucos(corners, ids)

        if rotated:
            rotated_image = cv.rotate(image, cv.ROTATE_180)
            return self.getTransform(rotated_image, corners, ids)
        
        for aruco_data in other_arucos:
            aruco_id = np.array([[aruco_data['id']]], dtype=np.int32)
            aruco_corners = aruco_data['points_image'].reshape((1, 4, 2))
            cv.aruco.drawDetectedMarkers(image=image, corners=[aruco_corners], ids=aruco_id, borderColor=(255,255,0))


        for aruco_data in aruco_list:
            aruco_id = np.array([[aruco_data['id']]], dtype=np.int32)
            aruco_corners = aruco_data['points_image'].reshape((1, 4, 2))
            
            cv.aruco.drawDetectedMarkers(image=image, corners=[aruco_corners], ids=aruco_id, borderColor=(0,0,255))

            center = projectCenter(aruco_data['points_image'])
            x, y, width, height = cv.boundingRect(aruco_data['points_image'])

            text = f"Search for {self.color} {self.shape}"
            color = self.colors_list[self.color]
            font = cv.FONT_HERSHEY_SIMPLEX
            scale = 1
            thickness = 1
            text_size, _ = cv.getTextSize(f'{text}', font, scale, thickness)
            text_origin = (int(center[0]-text_size[0]/2),int(center[1]+text_size[1]+3+height/2))
            cv.putText(image, text, org=text_origin, fontFace=font, fontScale=scale, color=color, thickness=thickness, lineType=cv.LINE_AA)

        if aruco_list:
            return True
        else:
            return False
        
    def getColorPixelInfo(self, coordinates_3d):
        pass
    
    def getPixelInfo(self, coordinates):

        if coordinates is not None:
            pass
        pass
    
## Other method has flaws. Adapted from: https://pyimagesearch.com/2016/03/21/ordering-coordinates-clockwise-with-python-and-opencv/
def sort_points_clockwise(pts):
   
    if pts.ndim == 2 and (pts.shape[1] == 2 or pts.shape[1] == 3): # a contour
        pass
    elif pts.ndim == 3 and (pts.shape[2] == 2 or pts.shape[2] == 3): # array of contours
        contour_list = []
        for contour in pts:
            contour_list.append(sort_points_clockwise(contour))
        return np.array(contour_list)
    
    # Handle 2D or 3D points by considering only the first two coordinates for sorting
    # pts_2d = pts[:, :2]

    # sort the points based on their x-coordinates
    xSorted = pts[np.argsort(pts[:, 0]), :]
    leftMost = xSorted[:2, :]
    rightMost = xSorted[2:, :]

    # now, sort the left-most coordinates according to their
    # y-coordinates so we can grab the top-left and bottom-left
    # points, respectively
    leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
    (tl, bl) = leftMost
    # now that we have the top-left coordinate, use it as an
    # anchor to calculate the Euclidean distance between the
    # top-left and right-most points; by the Pythagorean
    # theorem, the point with the largest distance will be
    # our bottom-right point
    D = dist.cdist(tl[np.newaxis], rightMost, "euclidean")[0]
    (br, tr) = rightMost[np.argsort(D)[::-1], :]
    # return the coordinates in top-left, top-right,
    # bottom-right, and bottom-left order
    ret = np.array([tl, tr, br, bl], dtype="float32")
    return ret 


"""
    Given a set of 3d coordinates, reescale them with image_size
    to match expected image dimensions
"""
def rescale_3d_points(points, img_shape):

    min_coord = np.min(points, axis=0)
    max_coord = np.max(points, axis=0)

    # img_shape is (rows, cols): index 1 is the X dimension and 0 the Y one
    scale_x = img_shape[1] / (max_coord[0] - min_coord[0])
    scale_y = img_shape[0] / (max_coord[1] - min_coord[1])
    
    # Scale with min to maintain square shapes as such
    scale = min(scale_x, scale_y)
    
    translation = [min_coord[0], min_coord[1]]
    translation += [0] * (len(points[0]) - len(translation))  # Adjust if is 3d points
    points_scaled = (points - translation) * scale
    
    return points_scaled, np.max(points_scaled, axis=0).astype(np.int32)

"""
    returns:
        Homograpy
        new image shape based on recomputed dimensions once the board is projected
"""
def aruco_board_transform(aruco_image_contours, aruco_3d_contours, board_3d_contours, img_shape):
    # M1 -> Transformation from 3d points aruco to image aruco coordinates

    aruco_3d_contours = np.concatenate(sort_points_clockwise(aruco_3d_contours), axis=0).astype("float32")
    aruco_image_contours = np.concatenate(sort_points_clockwise(aruco_image_contours), axis=0).astype("float32")

    # Compute the perspective transform matrix to transform between
    # 3d contours to image contours
    M1, _ = cv.findHomography(aruco_image_contours, aruco_3d_contours)

    board_image_contours = cv.perspectiveTransform(board_3d_contours.reshape(-1, 1, 2), np.linalg.inv(M1)).reshape(-1, 2)
    board_image_contours = sort_points_clockwise(board_image_contours)

    # Transform between board_image_points to new framed points. The warp target keeps the
    # board's METRIC aspect ratio (v1.4.1): map board_3d to a frame scaled by rescale_3d_points
    # (board aspect ~1.55) instead of stretching it to the full camera frame (1.778), which made
    # the top-view ~16% too wide. This matches the early-exit path (getTransform) so the warp
    # size is the same whether or not the pose is solved. Numerically neutral for the gaze->cell
    # result and the covariance (both projected per-axis through getPixelBoardNorm / the Jacobian);
    # it only fixes the visual proportions and keeps warped px ~isotropic for any future metric.
    _, (img_width, img_height) = rescale_3d_points(board_3d_contours, img_shape)
    image_points_framed = np.array([
        [0, 0],
        [img_width - 1, 0],
        [img_width - 1, img_height - 1],
        [0, img_height - 1]], dtype="float32")
    image_points_framed = sort_points_clockwise(image_points_framed)
    M2, _ = cv.findHomography(board_image_contours, image_points_framed)

    # ## Take into account board coordinates to reescale undistorted aruco contours, 
    # ## but then not include them into the homography as no board contour in image 
    # ## is matched
    # all_points_3d = np.concatenate((aruco_3d_contours,board_3d_contours), axis=0)
    # all_points_image_undistorted, (img_width, img_height, _) = rescale_3d_points(all_points_3d, img_shape)
    # aruco_3d_contours_undistorted = all_points_image_undistorted[:len(aruco_3d_contours)]

    # ## Computes homopgraphy between all image points and its coorespondand undistorted computed versions
    # M2, _ = cv.findHomography(aruco_image_contours, aruco_3d_contours_undistorted)
    
    return M2, int(img_width), int(img_height)
