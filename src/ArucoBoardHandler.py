
#!/usr/bin/env python3
# encoding: utf-8

import math
import copy

import cv2 as cv
import numpy as np
from scipy.spatial import distance as dist

import yaml
from yaml.loader import SafeLoader

from src.utils import projectCenter


"""
    Based on aruco markers detected, take white pixels and average the color correction
"""
def ARUCOColorCorrection(input_img):
    result = cv.cvtColor(input_img, cv.COLOR_BGR2LAB)
    avg_a = np.average(result[:, :, 1])
    avg_b = np.average(result[:, :, 2])
    result[:, :, 1] = result[:, :, 1] - ((avg_a - 128) * (result[:, :, 0] / 255.0) * 1.1)
    result[:, :, 2] = result[:, :, 2] - ((avg_b - 128) * (result[:, :, 0] / 255.0) * 1.1)
    result = cv.cvtColor(result, cv.COLOR_LAB2BGR)
    return result


    aruco_dictionary = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_ARUCO_ORIGINAL)
    corners, ids, rejectedImgPoints = cv.aruco.detectMarkers(input_img, aruco_dictionary)

    if ids is None:
        return input_img
    
    white_pixels = []
    for index, id in enumerate(ids):
        corner = corners[index][0]
        mask = np.zeros(input_img.shape[:2], dtype=np.uint8)
        cv.fillConvexPoly(mask, np.int32(corner), 255)
        white_pixels.append(input_img[mask == 255])

    white_pixels = np.concatenate(white_pixels, axis=0)
    average_white = np.mean(white_pixels, axis=0)

    # Takes all aruco, set tone to be average grey as if aruco had same white and black pixels...
    corrected_image = input_img*((255.0/2)/average_white)
    corrected_image = np.clip(corrected_image,0,255).astype(np.uint8)

    ## Noise reduction
    corrected_image = cv.bilateralFilter(corrected_image, d=9, sigmaColor=75, sigmaSpace=75)

    ## Adjust brightness
    gray = cv.cvtColor(corrected_image, cv.COLOR_BGR2GRAY)
    current_brightness = np.mean(gray)
    current_contrast = np.std(gray)
    expected_brightness = 190
    expected_contrast = 70
    alpha = expected_contrast/current_contrast # Contrast (more contrast alpha>1)
    beta = expected_brightness - current_brightness # brightness
    corrected_image = cv.convertScaleAbs(corrected_image, alpha=alpha, beta=beta)

    return corrected_image


def rotate_points_180_around_center(points, board_center):
    rotation_matrix = np.array([
        [-1, 0],
        [0, -1]
    ])

    translated_points = points - board_center
    rotated_points = np.dot(translated_points, rotation_matrix.T) + board_center

    return rotated_points



class ArucoBoardHandler:
    def __init__(self, arucoboard_cfg_path, colors_list, color, cameraMatrix = None, distCoeffs = None, shape = ''):

        self.colors_list = colors_list
        self.color = color
        self.shape = shape

        self.cameraMatrix = cameraMatrix
        self.distCoeffs = distCoeffs

        self.aruco_dictionary = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_ARUCO_ORIGINAL)
        self.aruco_config, self.board_poinst_3d, self.board_poinst_3d_rotated, self.aruco_side = self.parseCFGPanelData(arucoboard_cfg_path)
        

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

    def detectArucos(self, undistorted_frame):
        
        gray_image = cv.cvtColor(undistorted_frame, cv.COLOR_BGR2GRAY)  # transforms to gray level
        corners, ids, rejectedImgPoints = cv.aruco.detectMarkers(gray_image, self.aruco_dictionary)
        # cv.aruco.drawDetectedMarkers(image=undistorted_frame, corners=corners, ids=ids, borderColor=(0,0,255))

        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        for corner in corners:
            cv.cornerSubPix(gray_image, corner, winSize=(5, 5), zeroZone=(-1, -1), criteria=criteria)

        rotated_arucos = 0
        detected_aruco_list = []
        if ids is not None and ids.size > 0:
            for index, id in enumerate(ids):
                found = False
                for aruco_data in self.aruco_config:
                    if id == aruco_data['id']:
                        current_data = copy.deepcopy(aruco_data)
                        current_data.update({'points_image': corners[index][0]})
                        detected_aruco_list.append(current_data)
                        found = True  # ID encontrado
                        break
            
            rvecs, tvecs, _ = cv.aruco.estimatePoseSingleMarkers(corners, self.aruco_side, self.cameraMatrix, self.distCoeffs)
            ## Check rotation matrix, if is around 180 the board needs rotation :)
            
            for i, rvec in enumerate(rvecs):
                R, _ = cv.Rodrigues(rvec)  # Rotation matrix
                yaw = np.arctan2(R[1, 0], R[0, 0])
                if abs(yaw) > np.pi / 2:
                    rotated_arucos += 1
        
        if rotated_arucos >= len(self.aruco_config)*0.5:
            print(f"Need rotation! {rotated_arucos} out of {len(detected_aruco_list)}")
            need_rotation = True
        else:
            need_rotation = False
            
        return detected_aruco_list, need_rotation

    def getTransform(self, undistorted_frame):

        cv.imshow('getTransform::undistorted_frame', undistorted_frame)
        homography = None
        _, (warp_width, warp_height) = rescale_3d_points(self.board_poinst_3d, undistorted_frame.shape)
        detected_aruco_list, rotated = self.detectArucos(undistorted_frame)

        # Should detect all arucos?
        # Problem with board -> detect at least 60% of configured arucos
        if len(detected_aruco_list) > len(self.aruco_config)*0.5:
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

        return homography, int(warp_width), int(warp_height)


    def handleVisualization(self, image):
        aruco_list, rotated = self.detectArucos(image)

        if rotated:
            rotated_image = cv.rotate(image, cv.ROTATE_180)
            return self.getTransform(rotated_image)
            
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
    
    scale_x = img_shape[0] / (max_coord[0] - min_coord[0])
    scale_y = img_shape[1] / (max_coord[1] - min_coord[1])
    
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

    # Transform between board_image_points to new framed points
    # _, (img_width, img_height) = rescale_3d_points(board_3d_contours, img_shape)
    img_width, img_height = img_shape[1], img_shape[0]
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
