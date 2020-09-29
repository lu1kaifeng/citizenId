import math
from typing import List

import cv2


class ctpn_coordinate_pair:
    # (y1,x1,y2,x2)
    x1 = None
    y1 = None
    x2 = None
    y2 = None

    def __init__(self, t=(None, None, None, None)):
        self.y1 = t[0]
        self.x1 = t[1]
        self.y2 = t[2]
        self.x2 = t[3]


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


def get_rotation_angle(lines: List[ctpn_text_line]):
    return max(lines,key=lambda x:x.get_length()).get_angle()
    #Not Working
    # total_len = 0.0
    # for line in lines:
    #     total_len = total_len + line.get_length()
    # angle = 0.0
    # for line in lines:
    #     angle = angle  + line.get_angle() * line.get_length()
    # return angle / total_len
