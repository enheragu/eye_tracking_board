#!/usr/bin/env python3
# encoding: utf-8

import yaml
from yaml.loader import SafeLoader

import cv2 as cv
import numpy as np

def getBoardData(board_configuration):
    board_data_dict = {}
    with open(board_configuration) as file:
        data = yaml.load(file, Loader=SafeLoader)
        board_data_dict = {tuple(map(int, key.split(','))): value for key, value in data['board_config'].items()}
        board_size = data['board_size']      
        board_size_mm = data['board_size_mm']
    
    return board_size, board_size_mm, board_data_dict

def projectBoardConfig(cell_matrix, cell_width, cell_height, board_data_dict, display_image, colors_list):
    
    board_data_dict['cell_width'] = cell_width
    board_data_dict['cell_height'] = cell_height

    for i in range(len(cell_matrix)):
        for j in range(len(cell_matrix[i])):

            data = board_data_dict[(j,i)]
            x,y = cell_matrix[i][j]
            center = (int(x+cell_width/2), int(y+cell_height/2))
            text = f"{'Shape' if data[2] else 'Slot'}"
            text2 = f"{data[1]}"
            
            if len(board_data_dict[(j,i)]) <= 3:
                board_data_dict[(j,i)].append(cell_matrix[i][j])
            else:
                board_data_dict[(j,i)][3] = cell_matrix[i][j]

            if display_image is not None:
                color = colors_list[data[0]]
                font = cv.FONT_HERSHEY_SIMPLEX
                scale = 0.5
                thickness = 1
                text_size, _ = cv.getTextSize(f'{text}', font, scale, thickness)
                text_size2, _ = cv.getTextSize(f'{text2}', font, scale, thickness)
                text_origin = (int(center[0]-text_size[0]/2),int(center[1]-text_size[1]+3))
                text_origin2 = (int(center[0]-text_size2[0]/2),int(center[1]+text_size2[1]+3))
                cv.putText(display_image, text, org=text_origin, fontFace=font, fontScale=scale, color=color, thickness=thickness, lineType=cv.LINE_AA)
                cv.putText(display_image, text2, org=text_origin2, fontFace=font, fontScale=scale, color=color, thickness=thickness, lineType=cv.LINE_AA)
                cv.circle(display_image, center, radius=3, color=color, thickness=-1)

    return board_data_dict

def projectBoardMatrix(board_contour, board_size, display_image):
    x, y, w, h = cv.boundingRect(board_contour)

    cell_width = w // board_size[0]
    cell_height = h // board_size[1]

    cell_matrix = np.zeros((board_size[1], board_size[0], 2), dtype=int)
    for i in range(board_size[1]):
        for j in range(board_size[0]):
            x_corner = x + j * cell_width
            y_corner = y + i * cell_height
            
            cell_matrix[i,j] = [int(x_corner),int(y_corner)]

    if display_image is not None:
        for i in range(board_size[1]):
            for j in range(board_size[0]):
                x, y, w, h = cell_matrix[i, j]
                cv.rectangle(display_image, (x, y), (x + cell_width, y + cell_height), (0, 255, 0), 2)  # Dibujar un rectángulo verde con un grosor de 2 píxeles
                cv.putText(display_image, f'[{i},{j}]', org=(x+5,y+12), fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale=0.3, color=(0, 255, 0), thickness=1, lineType=cv.LINE_AA)
        
    return cell_matrix, cell_width, cell_height


def getCellIndex(pixel_coord, cell_info, cell_width, cell_height):
    x, y = pixel_coord
    
    cell_row = int((y - cell_info[0][0][1]) // cell_height)
    cell_col = int((x - cell_info[0][0][0]) // cell_width)
    
    if 0 <= cell_row < cell_info.shape[1] and \
       0 <= cell_col < cell_info.shape[0]:
        return int(cell_row), int(cell_col)
    else:
        return None, None