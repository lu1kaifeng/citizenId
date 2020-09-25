from __future__ import print_function
import cv2 as cv
import numpy as np
import argparse
import random as rng

rng.seed(12345)

inBlackVal = 0
inWhiteVal = 0
isGammaVal = 0
outBlackVal = 0
outWhiteVal = 0
def color_level(img):
    inBlack = np.array([inBlackVal, inBlackVal, inBlackVal], dtype=np.float32)
    inWhite = np.array([inWhiteVal, inWhiteVal, inWhiteVal], dtype=np.float32)
    inGamma = np.array([isGammaVal, isGammaVal, isGammaVal], dtype=np.float32)
    outBlack = np.array([outBlackVal,outBlackVal, outBlackVal], dtype=np.float32)
    outWhite = np.array([outWhiteVal, outWhiteVal, outWhiteVal], dtype=np.float32)

    img = np.clip((img - inBlack) / (inWhite - inBlack), 0, 255)
    img = (img ** (1 / inGamma)) * (outWhite - outBlack) + outBlack
    return np.clip(img, 0, 255).astype(np.uint8)

def thresh_callback(val):
    inBlackVal = val
    cv.imshow('Contours', color_level(src))


# Load source image
parser = argparse.ArgumentParser(description='Code for Finding contours in your image tutorial.')
parser.add_argument('--input', help='Path to input image.', default='HappyFish.jpg')
args = parser.parse_args()
src = cv.imread('D:\\citizenIdData\\Train_DataSet\\000d0143a5b442bdb8dd5d92e49f3b81.jpg')
if src is None:
    print('Could not open or find the image:', args.input)
    exit(0)
# Convert image to gray and blur it
src_gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
src_gray = cv.blur(src_gray, (3, 3))
# Create Window
source_window = 'Source'
cv.namedWindow(source_window)
cv.imshow(source_window, src)
max_thresh = 255
thresh = 100  # initial threshold
cv.createTrackbar('Canny Thresh:', source_window, thresh, max_thresh, thresh_callback)
thresh_callback(thresh)
cv.waitKey()
