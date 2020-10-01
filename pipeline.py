import re
from os import listdir
from os.path import isfile, join

import cv2

from CtpnPreprocessor import CtpnPreProcessor
from EmblemMatcher import EmblemMatcher
from card_crop import display_result
from label_processing import is_correct_crop
from perspective_zoom import perspective_zoomed_image
from tilt_align import ctpn_coordinate_pair, rotate_image, get_rotation_angle

preprocessor = CtpnPreProcessor()
emblem_matcher = EmblemMatcher()

interactive = True
path = r'D:\citizenIdData\Train_DataSet'
files = [f for f in listdir(path) if isfile(join(path, f)) and re.match('.*\\.jpg', f)]
#files=['0019905ad50e451e943d05ba5e090f72.jpg']
for file in files:
    img = cv2.imread(path + '\\' + file)
    sides, imgs = display_result(img, False)
    p0 = imgs[0]
    if len(imgs) is not 2:
        p1 = None
    else:
        p1 = imgs[1]
    if is_correct_crop(p0, p1):
        lines = preprocessor.get_text_lines(img)
        p0_lines = list(filter(lambda x: x.is_in_area(sides[0]), lines))
        p1_lines = list(filter(lambda x: x.is_in_area(sides[1]), lines))
        if interactive:
            for r in p0_lines:
                # (y1,x1,y2,x2)
                cv2.line(img, (r.x1, r.y1), (r.x2, r.y2), (255, 0, 0), 2)
            for r in p1_lines:
                # (y1,x1,y2,x2)
                cv2.line(img, (r.x1, r.y1), (r.x2, r.y2), (0, 255, 0), 2)
            cv2.imshow(file, img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        p0 = rotate_image(p0, get_rotation_angle(p0_lines))
        p0_lines = list(map(lambda x: x.to_context(sides[0]).rotate_by_angle(get_rotation_angle(p0_lines)), p0_lines))
        p0, p0_lines = perspective_zoomed_image(p0, p0_lines)
        p1 = rotate_image(p1, get_rotation_angle(p1_lines))
        p1_lines = list(map(lambda x: x.to_context(sides[1]).rotate_by_angle(get_rotation_angle(p1_lines)), p1_lines))
        p1, p1_lines = perspective_zoomed_image(p1, p1_lines)
        flip_back,front,front_lines,back,back_lines = emblem_matcher.side_orientation_recog(p0,p0_lines,p1,p1_lines,interactive)
        if flip_back:
            back = rotate_image(back, 180)
            back_lines = list(
                 map(lambda x: x.vertical_flip(back.shape[0],back.shape[1]), back_lines))
        if interactive:
            for r in back_lines:
                # (y1,x1,y2,x2)
                cv2.line(back, (r.x1, r.y1), (r.x2, r.y2), (255, 0, 0), 2)
            for r in front_lines:
                # (y1,x1,y2,x2)
                cv2.line(front, (r.x1, r.y1), (r.x2, r.y2), (0, 255, 0), 2)
            cv2.imshow('back', back)
            cv2.imshow('front', front)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    else:
        continue
