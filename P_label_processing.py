import re
from os import listdir
from os.path import isfile, join

import cv2
import numpy as np
import random as rng
import matplotlib.pyplot as plt
from rect_op import merge_bounding_boxes


def is_dimen_diff(p0, p1):
    return abs(p0.shape[0] - p1.shape[0]) > p0.shape[0] / 5 or abs(p0.shape[1] - p1.shape[1]) > p0.shape[1] / 5


def is_correct_crop( name: str, dir=r'D:\citizenIdData\train\\'):
    p0 = dir + name + '-0.jpg'
    p1 = dir + name + '-1.jpg'
    p0 = cv2.imread(p0)
    p1 = cv2.imread(p1)
    if p0 is None or p1 is None or is_dimen_diff(p0, p1):
        # if p0 is not None:
        #     cv2.imshow(p, p0)
        #     cv2.waitKey(0)
        #     cv2.destroyAllWindows()
        # if p1 is not None:
        #     cv2.imshow(p, p1)
        #     cv2.waitKey(0)
        #     cv2.destroyAllWindows()
        return True
    return False


path = r'D:\citizenIdData\Train_DataSet'
files = [f for f in listdir(path) if isfile(join(path, f)) and re.match('.*\\.jpg', f)]
i = 0
ii = 0
for file in files:
    if is_correct_crop( file):
        i = i + 1

    print(i)
