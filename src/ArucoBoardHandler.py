
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

class ArucoBoardHandler:
    def __init__(self, arucoboard_cfg_path, colors_list, color, shape = ''):

        self.colors_list = colors_list
        self.color = color
        self.shape = shape

        self.aruco_dictionary = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_ARUCO_ORIGINAL)
        self.aruco_config, self.board_poinst_3d = self.parseCFGPanelData(arucoboard_cfg_path)
        

    def parseCFGPanelData(self, arucoboard_cfg_path):
        aruco_3d_data = {}
        sheet_points_3d = None
        with open(arucoboard_cfg_path) as file:
            data = yaml.load(file, Loader=SafeLoader)
            board_size = data['board_size']
            board_position = data['board_position']
            aruco_side = data['aruco_side']


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
            

            board_points_3d = np.array([
                [board_position[0], board_position[1]], #, board_position[2]],                      # Esquina superior izquierda
                [board_position[0], board_position[1]+board_size[1]], #, board_position[2]],           # Esquina inferior izquierda
                [board_position[0]+board_size[0], board_position[1]], #, board_position[2]],           # Esquina superior derecha
                [board_position[0]+board_size[0], board_position[1]+board_size[1]] #, board_position[2]] # Esquina inferior derecha
            ], dtype=np.float32)

        
        return aruco_3d_data, board_points_3d

    def detectArucos(self, undistorted_frame):
        
        gray_image = cv.cvtColor(undistorted_frame, cv.COLOR_BGR2GRAY)  # transforms to gray level
        corners, ids, rejectedImgPoints = cv.aruco.detectMarkers(gray_image, self.aruco_dictionary)
        # cv.aruco.drawDetectedMarkers(image=undistorted_frame, corners=corners, ids=ids, borderColor=(0,0,255))

        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        for corner in corners:
            cv.cornerSubPix(gray_image, corner, winSize=(5, 5), zeroZone=(-1, -1), criteria=criteria)


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
                # if not found:
                #     print(f"ID {id} not in configuration: {self.aruco_config}")
        # else:
        #     print("No ids found in image")

        return detected_aruco_list

    def getTransform(self, undistorted_frame):
        homography = None
        _, (warp_width, warp_height) = rescale_3d_points(self.board_poinst_3d, undistorted_frame.shape)
        detected_aruco_list = self.detectArucos(undistorted_frame)

        
        aruco_corners_image = []
        aruco_corners_3d = []
        for detected_aruco in detected_aruco_list:
            aruco_corners_image.append(sort_points_clockwise(detected_aruco['points_image']))
            aruco_corners_3d.append(sort_points_clockwise(detected_aruco['points_3d']))

        if aruco_corners_image != [] and aruco_corners_3d != []:
            aruco_corners_image = np.array(aruco_corners_image, dtype=np.float32)
            aruco_corners_3d = np.array(aruco_corners_3d, dtype=np.float32)

            homography, warp_width, warp_height = aruco_board_transform(
                                    aruco_image_contours=aruco_corners_image,
                                    aruco_3d_contours=aruco_corners_3d,
                                    board_3d_contours=self.board_poinst_3d,
                                    img_shape=undistorted_frame.shape)

        return homography, int(warp_width), int(warp_height)


    def handleVisualization(self, image):

        aruco_list = self.detectArucos(image)
            
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

    def getPixelInfo(self, coordinates):
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
