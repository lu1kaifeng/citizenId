import re
from os import listdir
from os.path import isfile, join

import cv2
import numpy

from CtpnPreprocessor import CtpnPreProcessor
from EmblemMatcher import EmblemMatcher
from card_crop import display_result
from cnn_rnn_ctc.test_crnn import Test_CRNN
from homography_card_crop import HomographyPreprocessor
from label_processing import is_correct_crop
from perspective_zoom import perspective_zoomed_image
from text_seg import get_text_img
from tilt_align import ctpn_coordinate_pair, rotate_image, get_rotation_angle


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

interactive = True
# text_matcher = FrontTextMatcher()
ctc = Test_CRNN(batch_size=1,interactive=interactive)
hcc = HomographyPreprocessor(interactive=interactive)

path = r'D:\citizenIdData\Train_DataSet'
files = [f for f in listdir(path) if isfile(join(path, f)) and re.match('.*\\.jpg', f)]
#files=['00b4b438dbee4a35a42c8c9e23d22286.jpg']
for file in files:
    img = cv2.imread(path + '\\' + file)
    front,back = hcc.crop(img)
    if interactive:
        cv2.imshow('back', back)
        cv2.setMouseCallback('back', click_event)
        cv2.imshow('front', front)
        cv2.setMouseCallback('front', click_event)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    ctc.get_text_img(front, back)

