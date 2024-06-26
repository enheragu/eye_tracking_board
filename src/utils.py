
#!/usr/bin/env python3
# encoding: utf-8

import math

import cv2 as cv
import numpy as np


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

def buildMosaic(titles_list, images_list, rows, cols, margin = 15, margin_color=(255, 255, 255)):
    if len(images_list) != len(titles_list):
        raise ValueError("Number of titles and images provided to build mosaic do not match.")

    # Resize so all images have same height    
    max_height = max(img.shape[0] for img in images_list)
    max_width = max(img.shape[1] for img in images_list)
    
    mosaic_height = max_height * rows + (rows + 1) * margin
    mosaic_width = max_width * cols + (cols + 1) * margin

    # Crear el mosaico con el color de margen especificado
    mosaic = np.full((mosaic_height, mosaic_width, 3), margin_color, dtype=np.uint8)

    for i, (title, img) in enumerate(zip(titles_list, images_list)):
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

        mosaic[start_row:end_row, start_col:end_col] = img_with_title

    return mosaic

    
def add_title(image, title):
    # Copiar la imagen para no alterar la original
    img_with_title = image.copy()
    # Añadir el texto del título en la parte superior izquierda de la imagen
    cv.putText(img_with_title, title, (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv.LINE_AA)
    return img_with_title


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