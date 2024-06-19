#!/usr/bin/env python3
# encoding: utf-8

import math

import cv2 as cv
import numpy as np
from scipy.spatial import distance as dist

## Other method has flaws. Adapted from: https://pyimagesearch.com/2016/03/21/ordering-coordinates-clockwise-with-python-and-opencv/
def sort_points_clockwise(pts):

    if pts.ndim == 2 and pts.shape[1] == 2: # a contour
        pass
    elif pts.ndim == 3 and pts.shape[2] == 2: # array of contours
        contour_list = []
        for contour in pts:
            contour_list.append(sort_points_clockwise(contour))
            return np.array(contour_list)
            
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
    to matx expected image dimensions
"""
def rescale_3d_points(points, img_shape):

    min_x, min_y = np.min(points, axis=0)
    max_x, max_y = np.max(points, axis=0)
    
    scale_x = img_shape[0] / (max_x - min_x)
    scale_y = img_shape[1] / (max_y - min_y)
    
    # Scale with min to maintain square shapes as such
    scale = min(scale_x, scale_y)
    
    points_scaled = (points - [min_x, min_y]) * scale
    
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
    
    # Compute the perspective transform matrix and then apply it
    # (¡makes use of list of points, not contours!)
    M1, _ = cv.findHomography(aruco_3d_contours, aruco_image_contours)

    # Take into account board coordinates to reescale undistorted aruco contours, 
    # but then not include them into the homography as no board contour in image 
    # is matched
    all_points_3d = np.concatenate((aruco_3d_contours,board_3d_contours[0]), axis=0)
    all_points_image_undistorted, (img_width, img_height) = rescale_3d_points(all_points_3d, img_shape)
    aruco_3d_contours_undistorted = all_points_image_undistorted[:len(aruco_3d_contours)]

    # Computes homopgraphy between all image points and its coorespondand undistorted computed versions
    M2, _ = cv.findHomography(aruco_image_contours, aruco_3d_contours_undistorted)
    
    return M2, int(img_width), int(img_height)
