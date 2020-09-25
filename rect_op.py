import numpy as np
import cv2

def union(a, b):
    x = min(a[0], b[0])
    y = min(a[1], b[1])
    w = max(a[0] + a[2], b[0] + b[2]) - x
    h = max(a[1] + a[3], b[1] + b[3]) - y
    print(a, b, (x, y, w, h))
    return (x, y, w, h)


def intersection(a, b):
    x = max(a[0], b[0])
    y = max(a[1], b[1])
    w = min(a[0] + a[2], b[0] + b[2]) - x
    h = min(a[1] + a[3], b[1] + b[3]) - y
    if w < -2.5 or h < -2.5: return ()  # or (0,0,0,0) ?
    return (x, y, w, h)


def if_intersect_with_member(l, b):
    for lb in l:
        if intersection(lb, b) != ():
            return True
    return False


def merge_bounding_boxes(boxes: list) -> list:
    g1 = [boxes[0]]
    g2 = []
    for i in range(1, len(boxes)):
        if if_intersect_with_member(g1, boxes[i]):
            g1.append(boxes[i])
        else:
            g2.append(boxes[i])
    return [merge_group(g1), merge_group(g2)]


def merge_group(boxes: list) -> tuple:
    if len(boxes) == 0:
        return (0,0,0,0)
    target = boxes[0]
    for b in range(1, len(boxes)):
        target = union(target, boxes[b])
    return target
