
#!/usr/bin/env python3
# encoding: utf-8

import math
from tqdm import tqdm

import cv2 as cv
import numpy as np

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    ERROR = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

## log in terminal without affecting tqdm bar
def log(*args, **kwargs):
    tqdm.write(*args, **kwargs)   

from types import MappingProxyType
def print_tuple(tupla, indent=0):
    if all(isinstance(item, float) for item in tupla):
        log(f"({type(tupla).__name__}) - {tupla}")
        return

    log("")
    for index, item in enumerate(tupla):
        log(f"{'    ' * indent}· [{index}]: ", end="")
        
        if isinstance(item, dict) or isinstance(item, MappingProxyType):
            log("(dict)" if isinstance(item, dict) else "(mappingproxy)")
            nested_dict = dict(item) if isinstance(item, MappingProxyType) else item
            print_dict(nested_dict, indent + 1)
        elif isinstance(item, (list, tuple)):
            log(f"({type(item).__name__})")
            print_tuple(item, indent + 1)
        else:
            log(f"({type(item).__name__}) - {item}")

def print_named_dict(name, dictionary):
    log(f"{name}:")
    print_dict(dictionary, indent=1)

def print_dict(dictionary, indent=0):
    for key, value in dictionary.items():
        log(f"{'    '*indent}· {key}: ", end="")

        if isinstance(value, dict) or isinstance(value, MappingProxyType):
            log("(dict)" if isinstance(value, dict) else "(mappingproxy)")
            nested_dict = dict(value) if isinstance(value, MappingProxyType) else value
            print_dict(nested_dict, indent + 1)
        elif isinstance(value, (list, tuple)):
            print_tuple(value, indent + 1)
        else:
            log(f"({type(value).__name__}) - {value}")


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

def interpolate_points(contorno, num_puntos=10):
    # Obtener las coordenadas x e y de las esquinas del contorno
    x = contorno[:, 0, 0]
    y = contorno[:, 0, 1]

    # Crear un array de puntos interpolados entre las esquinas
    puntos_interpolados = []
    for i in range(len(x) - 1):
        x_interpolados = np.linspace(x[i], x[i + 1], num_puntos)
        y_interpolados = np.linspace(y[i], y[i + 1], num_puntos)
        puntos_interpolados.extend(zip(x_interpolados, y_interpolados))

    return np.array(puntos_interpolados)

def draw_rotated_text(img, text, pos, angle, scale=1, color=(255, 0, 0), thickness=2):
    font = cv.FONT_HERSHEY_SIMPLEX
    size = cv.getTextSize(text, font, scale, thickness)[0]
    
    text_img = np.zeros((size[1] + 10, size[0] + 10, 3), dtype=np.uint8)
    cv.putText(text_img, text, (5, size[1] + 5), font, scale, color, thickness, cv.LINE_AA)
    
    M = cv.getRotationMatrix2D((size[0] / 2 + 5, size[1] / 2 + 5), angle, 1)
    rotated_img = cv.warpAffine(text_img, M, (size[0] + 10, size[1] + 10))
    
    x, y = pos
    rows, cols, _ = rotated_img.shape
    for i in range(rows):
        for j in range(cols):
            if np.any(rotated_img[i, j]):
                img[y + i - rows // 2, x + j - cols // 2] = rotated_img[i, j]


#####################
#   Channel utils   #
#####################

def claheEqualization(channel):
    clahe = cv.createCLAHE(clipLimit=4.0, tileGridSize=(8,8))
    return clahe.apply(channel)

def centerHueChannel(hue_channel, new_center):
    lookUpTable = np.zeros(256, dtype=np.uint8)
    
    for i in range(len(lookUpTable)):
        lookUpTable[i] = (i+128-new_center) % 256

    return cv.LUT(hue_channel, lookUpTable)

def getMaskHue(hue, sat, intensity, h_ref, h_epsilon, s_margins = [5,255], v_margins = [60,215]):
    h_ref = int(h_ref/360.0*255.0)
    h_epsilon = int(h_epsilon/360.0*255.0)

    hue_new = centerHueChannel(hue, h_ref)
    hsv_new = cv.merge((hue_new, sat, intensity))

    res = cv.inRange(hsv_new, 
                        tuple([128-h_epsilon,s_margins[0],v_margins[0]]), 
                        tuple([128+h_epsilon,s_margins[1],v_margins[1]]))
    kernel = cv.getStructuringElement(cv.MORPH_CROSS, (3, 3))   # [[0,1,0], [1,1,1], [0,1,0]]
    
    # res = cv.erode(res,kernel, iterations = 2)
    res = cv.morphologyEx(res, cv.MORPH_OPEN, kernel,  iterations=2)
    res = cv.morphologyEx(res, cv.MORPH_CLOSE, kernel, iterations=2)
    
    return res



################################
#   Debug build mosaic utils   #
################################

def buildMosaic(titles_list, images_list, debug_data_list, rows, cols, margin = 15, margin_color=(255, 255, 255)):
    if len(images_list) != len(titles_list):
        raise ValueError("Number of titles and images provided to build mosaic do not match.")

    # Resize so all images have same height    
    max_height = max(img.shape[0] for img in images_list)
    max_width = max(img.shape[1] for img in images_list)
    
    mosaic_height = max_height * rows + (rows + 1) * margin
    mosaic_width = max_width * cols + (cols + 1) * margin

    # Crear el mosaico con el color de margen especificado
    mosaic = np.full((mosaic_height, mosaic_width, 3), margin_color, dtype=np.uint8)

    for i, (title, img, debug_data) in enumerate(zip(titles_list, images_list, debug_data_list)):
        row = i // cols
        col = i % cols
        start_row = row * (max_height + margin) + margin
        end_row = start_row + img.shape[0]
        start_col = col * (max_width + margin) + margin
        end_col = start_col + img.shape[1]

        roi = mosaic[start_row + margin:end_row - margin, start_col + margin:end_col - margin]

        if len(img.shape) < 3 or img.shape[2] == 1:
            img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)

        resized_img = resize_with_black_padding(img, (end_col - start_col, end_row - start_row))
        img_with_title = add_title(resized_img, title)
        img_with_title = add_debug_data(img_with_title, debug_data)

        mosaic[start_row:end_row, start_col:end_col] = img_with_title

    return mosaic

    
def add_title(image, title):
    # Copiar la imagen para no alterar la original
    img_with_title = image.copy()
    # Añadir el texto del título en la parte superior izquierda de la imagen
    cv.putText(img_with_title, title, (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv.LINE_AA)
    return img_with_title


def add_debug_data(image, debug_data):
    img_with_data = image.copy()
    if debug_data is not None:
        for index, text in enumerate(debug_data):
            font = cv.FONT_HERSHEY_SIMPLEX
            scale = 0.4
            thickness = 1
            text_size, _ = cv.getTextSize(text, font, scale, thickness)
            text_width, text_height = text_size
            x = 10
            y = 60 + text_height + (10 + text_height)*index
            cv.putText(img_with_data, text, (x, y), font, scale, (255,255,0), thickness)
    return img_with_data

def resize_with_black_padding(image, new_size):
    h, w = image.shape[:2]
    new_w, new_h = new_size

    scale_w = new_w / w
    scale_h = new_h / h

    if scale_w > scale_h:
        new_w = int(w * scale_h)
    else:
        new_h = int(h * scale_w)

    resized_img = cv.resize(image, (new_w, new_h), interpolation=cv.INTER_AREA)

    padded_img = np.zeros((new_size[1], new_size[0], 3), dtype=np.uint8)
    x_offset = (new_size[0] - new_w) // 2
    y_offset = (new_size[1] - new_h) // 2
    padded_img[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized_img

    return padded_img


def getMosaic(capture_idx, last_capture_idx, fps, frame_width, frame_height, titles_list, images_list, rows, cols, resize = 1, debug_data_list = None):

    if debug_data_list is None:
        debug_data_list = [None*len(images_list)]

    for index, resized_image in enumerate(images_list):
        images_list[index] = cv.resize(resized_image, (int(frame_width*resize), int(frame_height*resize)))

    mosaic = buildMosaic(titles_list=titles_list, 
                images_list=images_list, 
                rows=rows, cols=cols,
                debug_data_list=debug_data_list)

    text = f'Frame: {capture_idx}/{last_capture_idx}'
    font = cv.FONT_HERSHEY_SIMPLEX
    scale = 0.4
    thickness = 1
    text_size, _ = cv.getTextSize(text, font, scale, thickness)
    text_width, text_height = text_size
    x = mosaic.shape[1] - text_width - 1  # 10 pixel margin
    y = text_height + 1  # 10 pixel margin
    cv.putText(mosaic, text, (x, y), font, scale, color=(0,0,0), thickness=thickness)

    cv.putText(mosaic, f"FPS: {fps:.1f}", org=(3, 10),
        fontFace=font, fontScale=scale, color=(0,0,0), thickness=thickness)
    

    return mosaic