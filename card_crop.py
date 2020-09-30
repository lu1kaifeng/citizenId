import re
from os import listdir
from os.path import isfile, join

import cv2
import numpy as np
import random as rng
import matplotlib.pyplot as plt
from rect_op import merge_bounding_boxes
from tilt_align import ctpn_coordinate_pair


def adjust_gamma(image, gamma=1.0):
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)


def is_in_rect(rect, collection):
    for c in collection:
        if (c[0] + c[2] > rect[0] > c[0] and
            c[1] + c[3] > rect[1] > c[1]) and \
                (c[0] + c[2] > rect[0] + rect[2] > c[0] and
                 c[1] + c[3] > rect[1] + rect[3] > c[1]):
            return True
    return False


PADDING = 10


def add_padding_to_bounding_rect(rect: tuple) -> tuple:
    return rect[0] - PADDING , rect[1] - PADDING, rect[2] +2 *PADDING , rect[3] + 2*PADDING


def display_result(img, interactive=True):

    edge = cv2.Canny(cv2.erode(img, np.ones((7, 7), np.uint8)), 15, 30)
    fig, ax = plt.subplots(1, figsize=(12, 8))

    # Find contours
    _, contours, hierarchy = cv2.findContours(edge, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # Draw contours
    drawing = np.zeros((edge.shape[0], edge.shape[1], 3), dtype=np.uint8)

    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    rect = []
    for i in range(len(contours)):
        color = (255, 255, 255)
        cv2.drawContours(drawing, contours, i, color, 2, cv2.LINE_8, hierarchy, 0)
        rect.append(cv2.boundingRect(contours[i]))
    rect = list(sorted(rect, key=lambda x: x[2] * x[3], reverse=True))
    largest = rect[0][2] * rect[0][3]
    rect = list(filter(lambda x: not (x[0] < 10 and x[1] < 10), rect))
    rect = list(filter(lambda x: (x[2] * x[3]) <= (largest * 0.75), rect))
    rect = list(filter(lambda x: not is_in_rect(x, rect), rect))
    rect1 = merge_bounding_boxes(rect)
    i = 0
    imgs = []
    rectObj = []
    for r in rect1:
        cv2.rectangle(drawing, (r[0], r[1]), (r[0] + r[2], r[1] + r[3]), (0, 255, 0), 2)
        pair = ctpn_coordinate_pair()
        pair.x1 = r[0]
        pair.y1=r[1]
        pair.x2 = r[0] + r[2]
        pair.y2 = r[1] + r[3]
        rectObj.append(pair)
        imgs.append(img[r[1]:r[1] + r[3], r[0]:r[0] + r[2]])
        i = i + 1

    for r in rect1:
        cv2.rectangle(drawing, (r[0], r[1]), (r[0] + r[2], r[1] + r[3]), (0, 0, 255), 2)

        for r in rect:
            cv2.rectangle(drawing, (r[0], r[1]), (r[0] + r[2], r[1] + r[3]), (255, 0, 0), 2)
    if interactive:
        cv2.imshow('imgg', drawing)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return rectObj,imgs


# path = r'D:\citizenIdData\Train_DataSet'
# path1 = '00bb30d5ac0448a192ba2bfd6d5c99a2.jpg'
# files = [f for f in listdir(path) if isfile(join(path, f)) and re.match('.*\\.jpg', f)]
# for file in files:
#     display_result(cv2.imread(path + '\\' + file), file,interactive=False)
