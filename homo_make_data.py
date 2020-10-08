import re
from os import listdir
from os.path import isfile, join

import cv2
import numpy as np

from homography_card_crop import HomographyPreprocessor


def arrange_lines(img: np.ndarray):
    height = img.shape[0]
    line_height = int(height / 3)
    return np.concatenate(
        (img[0:line_height, :], img[line_height:line_height * 2, :], img[line_height * 2:line_height * 3, :]), axis=1)


def click_event(event, x, y, flags, params):
    # checking for left mouse clicks
    if event == cv2.EVENT_LBUTTONDOWN:
        # displaying the coordinates
        # on the Shell
        print(x, ' ', y)

        # checking for right mouse clicks
    if event == cv2.EVENT_RBUTTONDOWN:
        # displaying the coordinates
        # on the Shell
        print(x, ' ', y)


interactive = False
# text_matcher = FrontTextMatcher()
# ctc = Test_CRNN(batch_size=1,interactive=interactive)
hcc = HomographyPreprocessor(interactive=interactive)

path = r'D:\citizenIdData\Train_DataSet'
files = [f for f in listdir(path) if isfile(join(path, f)) and re.match('.*\\.jpg', f)]
#files = ['0e8df807a2a641b2851114bed1cda7a0.jpg']
for file in files:
    img = cv2.imread(path + '\\' + file)
    front, back = hcc.crop(img)
    if front is not None and back is not None:
        if interactive:
            cv2.imshow('back', back)
            cv2.setMouseCallback('back', click_event)
            cv2.imshow('front', front)
            cv2.setMouseCallback('front', click_event)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        myDict = {
            "pd": back[204:240, 169:388],
            "period": back[245:280, 168:363],
            "name": front[48:84, 71:153],
            "sex": front[86:119, 76:113],
            "ethnic": front[84:117, 176:245],
            "birth": front[112:146, 74:243],
            "address": arrange_lines(front[151:225, 78:300]),
            "id": front[230:272, 123:385]
        }
        for k, v in myDict.items():
            cv2.imwrite(r'D:\citizenIdData\ctc_train' + '\\' + file + '-' + k + '.jpg', v)
