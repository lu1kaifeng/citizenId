import math
from typing import List

import cv2
import numpy as np

class ctpn_coordinate_pair:
    # (y1,x1,y2,x2)
    x1 = None
    y1 = None
    x2 = None
    y2 = None
    odd = 0.0

    def __init__(self, t=(None, None, None, None),odd=0.0):
        self.y1 = t[1]
        self.x1 = t[0]
        self.y2 = t[3]
        self.x2 = t[2]
        self.odd = odd

    def get_angle(self):
        return -math.atan((self.y1 - self.y2) / (self.x2 - self.x1)) * 180 / math.pi

    def get_length(self):
        return math.sqrt(math.pow(self.y2 - self.y1, 2) + math.pow(self.x2 - self.x1, 2))

    def is_in_area(self,area):
        if (area.x2 > self.x1 > area.x1 and
            area.x2 > self.x2 > area.x1) and \
                (area.y2 > self.y1 > area.y1 and
                 area.y2 > self.y2 > area.y1):
            return True
        else:
            return False

    def rotate_by_angle(self,angle: float):
        # inverse matrix of simple rotation is reversed rotation.
        M_inv = cv2.getRotationMatrix2D((100 / 2, 300 / 2), angle, 1)

        # points
        points = np.array([[self.x1, self.y1],
                           [self.x2, self.y2]])
        # add ones
        ones = np.ones(shape=(len(points), 1))

        points_ones = np.hstack([points, ones])

        points = M_inv.dot(points_ones.T).T
        pair = ctpn_coordinate_pair()
        pair.x1 = int(points[0][0])
        pair.y1 = int(points[0][1])
        pair.x2 = int(points[1][0])
        pair.y2 = int(points[1][1])
        return pair

    def to_context(self,context):
        pair = ctpn_coordinate_pair()
        pair.x1 = self.x1 - context.x1
        pair.y1 = self.y1 - context.y1
        pair.x2 = self.x2 - context.x1
        pair.y2 = self.y2 - context.y1
        return pair

    def vertical_flip(self,image_height,image_width):
        pair = ctpn_coordinate_pair()
        pair.x1 = image_width - self.x1
        pair.y1 = image_height - self.y1
        pair.x2 = image_width - self.x2
        pair.y2 = image_height - self.y2
        return pair

class ctpn_text_line:
    elem = []
    first = None
    last = None

    def __init__(self, line):
        self.elem = line

    def get_cross_line(self):
        y_start = (self.elem[0].y2 + self.elem[0].y1) / 2
        y_end = (self.elem[len(self.elem) - 1].y2 + self.elem[len(self.elem) - 1].y1) / 2
        coord = ctpn_coordinate_pair()
        coord.y1 = int(y_start)
        coord.y2 = int(y_end)
        coord.x1 = int(self.elem[0].x1)
        coord.x2 = int(self.elem[len(self.elem) - 1].x1)
        return coord

    def get_angle(self):
        coord = self.get_cross_line()
        return -math.atan((coord.y1 - coord.y2) / (coord.x2 - coord.x1)) * 180 / math.pi

    def get_length(self):
        coord = self.get_cross_line()
        return math.sqrt(math.pow(coord.y2 - coord.y1, 2) + math.pow(coord.x2 - coord.x1, 2))


def get_lines(boxes: list):
    boxes = map(lambda x: ctpn_coordinate_pair(x), boxes)
    boxes = sorted(boxes, key=lambda x: (x.y1 + x.y2) / 2)
    lines = []
    while len(boxes) is not 0:
        line = []
        lines.append(line)
        for b in boxes[:]:
            if len(line) is 0:
                line.append(b)
                boxes.remove(b)
            else:
                if abs(((line[len(line) - 1].y1 + line[len(line) - 1].y2) / 2) - ((b.y1 + b.y2) / 2)) < (
                        line[len(line) - 1].y2 - line[len(line) - 1].y1):
                    line.append(b)
                    boxes.remove(b)
                else:
                    boxes.remove(b)
                    break
    return list(map(lambda x: ctpn_text_line(x), filter(lambda x: len(x) > 3, lines)))


def get_rotation_angle(lines: List[ctpn_coordinate_pair]):
    total_len = 0.0
    for line in lines:
        total_len = total_len + math.pow(line.get_length(),3)*line.odd
    angle = 0.0
    for line in lines:
        angle = angle  + line.get_angle() * math.pow(line.get_length(),3)*line.odd
    return angle / total_len

def rotate_image(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result