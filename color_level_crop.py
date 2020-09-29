import re
from os import listdir
from os.path import isfile, join

import cv2
import numpy as np
import random as rng
import matplotlib.pyplot as plt
from rect_op import merge_bounding_boxes

PADDING = 5


def color_level(img):
    inBlack = np.array([1, 1, 1], dtype=np.float32)
    inWhite = np.array([195, 195, 195], dtype=np.float32)
    inGamma = np.array([0.01, 0.01, 0.01], dtype=np.float32)
    outBlack = np.array([1, 1, 1], dtype=np.float32)
    outWhite = np.array([255, 255, 255], dtype=np.float32)

    img = np.clip((img - inBlack) / (inWhite - inBlack), 0, 255)
    img = (img ** (1 / inGamma)) * (outWhite - outBlack) + outBlack
    return np.clip(img, 0, 255).astype(np.uint8)


def is_in_rect(rect, collection):
    for c in collection:
        if c[0] + c[2] - PADDING > rect[0] > c[0] + PADDING and c[1] + c[3] - PADDING > rect[1] > c[1] + PADDING:
            return True
    return False


def add_padding_to_bounding_rect(rect: tuple) -> tuple:
    return rect[0] - PADDING, rect[1] - PADDING, rect[2] + PADDING, rect[3] + PADDING


def display_result(p: str, name: str, dir=r'D:\citizenIdData\train\\'):
    img = cv2.imread(p)
    img = color_level(img)
    img = cv2.pyrMeanShiftFiltering(img, 50, 100)
    edge = cv2.Canny(cv2.erode(img, np.ones((9, 9), np.uint8)), 15, 30)

    # Find contours
    _, contours, hierarchy = cv2.findContours(edge, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # Draw contours
    drawing = np.zeros((edge.shape[0], edge.shape[1], 3), dtype=np.uint8)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    rect = []
    for i in range(len(contours)):
        color = (255, 255, 255)
        cv2.drawContours(drawing, contours, i, color, 2, cv2.LINE_8, hierarchy, 0)
        rect.append(add_padding_to_bounding_rect(cv2.boundingRect(contours[i])))

    i = 0
    for r in rect:
        cv2.rectangle(drawing, (r[0], r[1]), (r[0] + r[2], r[1] + r[3]), (0, 255, 0), 2)
        cv2.imwrite(dir + name + r'-' + str(i) + '.jpg', img[r[1]:r[1] + r[3], r[0]:r[0] + r[2]])
        i = i + 1
    cv2.imshow(p+' img', img)
    cv2.imshow(p, drawing)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


path = r'D:\citizenIdData\Train_DataSet'
files = [f for f in listdir(path) if isfile(join(path, f)) and re.match('.*\\.jpg', f)]
for file in files:
    display_result(path + '\\' + file, file)
