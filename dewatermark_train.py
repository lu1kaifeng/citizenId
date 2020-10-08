import re
from os import listdir
from os.path import isfile, join

import cv2

from homography_card_crop import HomographyPreprocessor

hcc = HomographyPreprocessor(interactive=False)
path = r'D:\citizenIdData\Train_DataSet'
files = [f for f in listdir(path) if isfile(join(path, f)) and re.match('.*\\.jpg', f)]
#files=['00b4b438dbee4a35a42c8c9e23d22286.jpg']
for file in files:
    img = cv2.imread(path + '\\' + file)
    front,back = hcc.crop(img)
    cv2.imwrite(r'D:\citizenIdData\train'+'\\'+file+'-front.jpg',front)
    cv2.imwrite(r'D:\citizenIdData\train' + '\\' + file + '-back.jpg', back)