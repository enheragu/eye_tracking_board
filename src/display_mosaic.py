#!/usr/bin/env python3
# encoding: utf-8

"""
    Debugging tool to present a set of images in just one imshow as a mosaic
"""

import cv2 as cv
import numpy as np

def imshowMosaic(titles_list, images_list, rows, cols, window_name = None, margin = 10, margin_color=(255, 255, 255)):
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

    cv.imshow(window_name, mosaic)

    
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