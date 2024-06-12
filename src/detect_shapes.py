#!/usr/bin/env python3
# encoding: utf-8

import math

import cv2 as cv
import numpy as np

from src.channel_utils import getMaskHue, claheEqualization
from src.display_mosaic import imshowMosaic

def projectCenter(contour):
    center = None
    M = cv.moments(contour)
    if M["m00"] != 0:
        # Compute centroid coordinates
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        center_x = int((M['m10'] / M['m00']))
        center_y = int((M['m01'] / M['m00']))
        center = (center_x, center_y)
    return center


def checkShape(contour, area_filter = [900,10000]):
    
    shapes = {'triangle': {'sides': 3, 'aspect_ratio': None, 'compacity': [0.5,0.7]},
              'square': {'sides': 4, 'aspect_ratio': [0.8,1.1], 'compacity': [0.71,0.9]},
              'rectangle': {'sides': 4, 'aspect_ratio': None, 'compacity': None}, #[0.71,0.9]}, # if not square
              'circle': {'sides': None, 'aspect_ratio': None, 'compacity': [0.91, 1]},
              'hexagon': {'sides': 6, 'aspect_ratio': None, 'compacity': None},
              'trapezoid': {'sides': 4, 'aspect_ratio': None, 'compacity': None}
            }
    
    perimeter = cv.arcLength(contour, True)
    if not perimeter > 0.1:
        return None, None

    approximate = cv.approxPolyDP(contour, .04 * perimeter, True)
    area = cv.contourArea(contour)

    if area < area_filter[0] or area > area_filter[1]:
        return None, approximate

    x, y, w, h = cv.boundingRect(contour)
    aspect_ratio = float(w) / h

    compacity = 4 * np.pi * area / (perimeter * perimeter)

    for shape, data in shapes.items():
        if data['sides'] is not None and len(approximate) != data['sides']:
            continue

        if data['aspect_ratio'] is not None and not (data['aspect_ratio'][0] < aspect_ratio < data['aspect_ratio'][1]):
            continue
        
        if data['compacity'] is not None and not data['compacity'][0] < compacity < data['compacity'][1]:
            continue

        return shape, approximate
    
    
    return None, approximate


def detectBoardContour(image, display_image, color = (255,255,0)):
    global margin_color

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
            if display_image is not None:
                display_image = cv.drawContours(display_image, [board_contour], -1, color, 2)
            break
            
            
    # cv.imshow(f'border_edges', edge_image)
    # cv.imshow(f'border_contour', display_image)
    return board_contour


def detectColorSquares(image, color_dict, colors_list, display_image):
    
    contours_filtered = dict()

    hue, sat, intensity = cv.split(cv.cvtColor(image, cv.COLOR_BGR2HSV_FULL))
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    debug_mask = {}
    for color, h_ref in color_dict.items():
        res = getMaskHue(hue, sat, intensity, h_ref['h'], h_ref['eps'])
        edge_image = cv.Canny(res, threshold1=50, threshold2=200)

        contours, hierarchy = cv.findContours(edge_image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        squares = list()
        for contour in contours:
            shape, approx = checkShape(contour, area_filter=[1000, math.inf])
            if shape == 'square':
                squares.append(approx)
        
        if display_image is not None:
            display_image = cv.drawContours(display_image, squares, -1, colors_list[color], 2)
        contours_filtered[color] = squares

        # cv.imshow(f'{color}_contour', edge_image)
        debug_mask[f'{color}_contour'] = edge_image
    
    if display_image is not None:
        imshowMosaic(debug_mask.keys(), debug_mask.values(), 2, 2, 'Contour Masks')
    
    return contours_filtered


def isSlot(image, bounding_box, color_item, color, display_image):

    ## Detect type fo shape -> Not possible with this images :(
    # x, y, w, h = cv.boundingRect(bounding_box)
    # cut_w = int(w * 0.95)  # Reducir el ancho al 80%
    # cut_h = int(h * 0.95)  # Reducir la altura al 80%
    # cut_x = x + int((w - cut_w) / 2)  # Centrar horizontalmente el recorte
    # cut_y = y + int((h - cut_h) / 2)  # Centrar verticalmente el recorte

    # area_interest = image[cut_y:cut_y+cut_h, cut_x:cut_x+cut_w]
    # area_interest = cv.cvtColor(area_interest, cv.COLOR_BGR2GRAY)
    
    # edge_image = cv.Canny(area_interest, threshold1=20, threshold2=180)
    # contours, hierarchy = cv.findContours(edge_image, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    
    # current_shape = None
    # for contour in contours:
    #     shape, approx = checkShape(contour, area_filter=[30, math.inf])
    #     if shape is not None:
    #         current_shape = shape
        
    #     area_interest = cv.drawContours(area_interest, [approx], -1, color, 2)
    # cv.imshow(f'detectShape:contour', area_interest)
    


    ## Detect if theres shape or slot
    is_slot = False
    hue, sat, intensity = cv.split(cv.cvtColor(image, cv.COLOR_BGR2HSV_FULL))
    center = projectCenter(bounding_box)
    
    region_size = 3

    x1 = (max(0, center[0]-region_size), max(0, center[1]-region_size))
    x2 = (min(image.shape[0], center[0]-region_size), min(image.shape[1]-1, center[1]-region_size))
    
    mask = np.zeros_like(hue)
    cv.rectangle(mask, x1, x2, (255, 255, 255), -1)

    # Detect white background if slot
    res = getMaskHue(hue, sat, intensity, color_item['h'], 180, s_margins=[0,20])
    region_img = cv.bitwise_and(res, mask)

    background_detected = np.sum(region_img)
    
    if background_detected:
        text = f'Slot'
        is_slot = True
    else:
        text = f'Shape'

    
    if display_image is not None:
        font = cv.FONT_HERSHEY_SIMPLEX
        scale = 0.5
        thickness = 1
        text_size, _ = cv.getTextSize(f'{text}', font, scale, thickness)
        text_origin = (int(center[0]-text_size[0]/2),int(center[1]-text_size[1]))
        cv.putText(display_image, text, org=text_origin, fontFace=font, fontScale=scale, color=color, thickness=thickness, lineType=cv.LINE_AA)
        cv.circle(display_image, center, radius=3, color=color, thickness=-1)

    return is_slot, center