#!/usr/bin/env python3
# encoding: utf-8

import math

import cv2 as cv
import numpy as np
from scipy.spatial import distance as dist

## Other method has flaws. Adapted from: https://pyimagesearch.com/2016/03/21/ordering-coordinates-clockwise-with-python-and-opencv/
def sort_points_clockwise(pts):
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
    return np.array([tl, tr, br, bl], dtype="float32")

## Adapted from: https://pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example/
def four_point_transform(pts):
    pts = np.array(pts, dtype=np.float32)

    rect = sort_points_clockwise(pts)
    (tl, tr, br, bl) = rect
    # compute the width  and height of the new image:
    # maximum distance between bottom/right and bottom/left
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype = "float32")
    # compute the perspective transform matrix and then apply it
    M = cv.getPerspectiveTransform(rect, dst)

    return M, int(maxWidth), int(maxHeight)

## Compute the four points transform of the points given an extra margin 
# to add more info of the image
def margin_four_point_transform(pts, img_shape = None, margin = 20):
    pts = np.array(pts, dtype=np.float32)

    M1, maxWidth, maxHeight = four_point_transform(pts)

    margin_dst1 = np.array([
        [-margin, -margin],
        [maxWidth - 1 + margin, -margin],
        [maxWidth - 1 + margin, maxHeight - 1 + margin],
        [-margin, maxHeight - 1 + margin]], dtype = "float32")
    
    pts_2 = cv.perspectiveTransform(np.array([margin_dst1]), np.linalg.inv(M1))[0]
    return four_point_transform(pts_2)

## Compute transform for whole image based on given points
def image_four_point_transform(pts, img_shape):
    ## Corrects all image?¿
    pts = np.array(pts, dtype=np.float32)

    M1, maxWidth, maxHeight = four_point_transform(pts)
    # new source: image corners
    # Transform those new points
    corners = np.array([[0, img_shape[0]],[0, 0],[img_shape[1], 0],[img_shape[1], img_shape[0]]])
    corners = sort_points_clockwise(corners)
    corners_tranformed = cv.perspectiveTransform(np.array([corners.astype("float32")]), M1)
    
    x_min = corners_tranformed[0][0][0]
    y_min = corners_tranformed[0][0][1]
    x_max = corners_tranformed[0][2][0]
    y_max = corners_tranformed[0][2][1]
    
    width = int(np.ceil(x_max - x_min))
    height = int(np.ceil(y_max - y_min))

    adjusted_dst = np.array([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1]], dtype="float32")
    adjusted_dst = sort_points_clockwise(adjusted_dst)
    print(f'{corners_tranformed = }')
    print(f'{x_min = }; {y_min = }; {x_max = }; {y_max = }')
    print(f'{width = }; {height = }')
    print(f'{adjusted_dst = }')

    M2 = cv.getPerspectiveTransform(corners.astype("float32"),adjusted_dst.astype("float32"))
    return M2, width, height
