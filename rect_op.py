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
    if len(l) is 0:
        return False
    for lb in l:
        if intersection(lb, b) != ():
            return True
    return False


def total_area(boxes: list):
    s = 0
    for i in boxes:
        s = s + i[2] * i[3]
    return s


def merge_bounding_boxes(boxes: list) -> list:
    g1 = [boxes[0]]
    g2 = []
    for i in range(1, len(boxes)):
        if if_intersect_with_member(g1, boxes[i]):
            g1.append(boxes[i])
        else:
            g2.append(boxes[i])
    if total_area(g1) <= total_area(g2):
        a = g1
        b = g2
    else:
        a = g2
        b = g1
    a_group = merge_group(a)
    b_group = merge_group(b)
    for i in a[:]:
        if if_intersect_with_member([a_group], i) and if_intersect_with_member([b_group], i):
            a.remove(i)
            a_group = merge_group(a)
            b_group = merge_group(b)
    return [merge_group(a), merge_group(b)]


def merge_group(boxes: list) -> tuple:
    if len(boxes) == 0:
        return (0,0,0,0)
    target = boxes[0]
    for b in range(1, len(boxes)):
        target = union(target, boxes[b])
    return target
