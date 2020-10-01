from typing import List

import numpy as np
import cv2

from card_crop import is_in_rect
from rect_op import merge_bounding_boxes
from tilt_align import ctpn_coordinate_pair

PADDING = 10
HEIGHT = 300
WIDTH = 440

def perspective_zoomed_image(img: np.ndarray, lines: List[ctpn_coordinate_pair], interactive=True):
    rows, cols, ch = img.shape
    # fast = cv2.FastFeatureDetector_create()
    # kp = fast.detect(img, None)
    # cv2.drawKeypoints(img, kp, color=(255, 0, 0),outImage=img)
    # cv2.imshow('x',img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    pts1 = np.float32(
        [[PADDING, PADDING], [cols - PADDING, PADDING], [PADDING, rows - PADDING], [cols - PADDING, rows - PADDING]])
    pts2 = np.float32([[0, 0], [WIDTH, 0], [0, HEIGHT], [WIDTH, HEIGHT]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)

    def get_points(x, y):
        px = (matrix[0][0] * x + matrix[0][1] * y + matrix[0][2]) / (
            (matrix[2][0] * x + matrix[2][1] * y + matrix[2][2]))
        py = (matrix[1][0] * x + matrix[1][1] * y + matrix[1][2]) / (
            (matrix[2][0] * x + matrix[2][1] * y + matrix[2][2]))
        return int(px), int(py)

    for l in lines:
        l.x1, l.y1 = get_points(l.x1, l.y1)
        l.x2, l.y2 = get_points(l.x2, l.y2)
    return cv2.warpPerspective(img, matrix, (WIDTH, HEIGHT)),lines
