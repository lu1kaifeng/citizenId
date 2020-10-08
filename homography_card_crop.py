import re
from os import listdir
from os.path import isfile, join

import numpy as np
import cv2
from matplotlib import pyplot as plt


class HomographyPreprocessor:
    MIN_MATCH_COUNT = 10

    def __init__(self,interactive=False):
        self.interactive = interactive
        self.sift = cv2.xfeatures2d.SIFT_create()
        self.front_tem = cv2.imread('./data/front.jpg', 0)
        self.front_tem_kp, self.front_tem_des = self.sift.detectAndCompute(self.front_tem, None)
        self.back_tem = cv2.imread('./data/back.jpg', 0)
        self.back_tem_kp, self.back_tem_des = self.sift.detectAndCompute(self.back_tem, None)

    def _detect(self, kp, des, tem, train_img):

        input_kp, input_des = self.sift.detectAndCompute(train_img, None)

        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)

        flann = cv2.FlannBasedMatcher(index_params, search_params)

        matches = flann.knnMatch(des, input_des, k=2)

        # store all the good matches as per Lowe's ratio test.
        good = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good.append(m)

        if len(good) > self.MIN_MATCH_COUNT:
            src_pts = np.float32([kp[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst_pts = np.float32([input_kp[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

            M, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
            matches_mask = mask.ravel().tolist()

            h, w = tem.shape
            # pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
            # dst = cv2.perspectiveTransform(pts, M)
            #
            # train_img = cv2.polylines(train_img, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)
            card = cv2.warpPerspective(train_img, M, (w, h))
        else:
            print("Not enough matches are found - %d/%d" % (len(good), self.MIN_MATCH_COUNT))
            matches_mask = None
            return None

        draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                           singlePointColor=None,
                           matchesMask=matches_mask,  # draw only inliers
                           flags=2)

        img3 = cv2.drawMatches(tem, kp, train_img, input_kp, good, None, **draw_params)
        if self.interactive:
            cv2.imshow('img3', img3)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        return card

    def crop(self, input_img):
# return cv2.cvtColor(self._detect(self.front_tem_kp, self.front_tem_des, self.front_tem, input_img), cv2.COLOR_BGR2GRAY), cv2.cvtColor(self._detect(
#     self.back_tem_kp,
#     self.back_tem_des,
#     self.back_tem,
#     input_img), cv2.COLOR_BGR2GRAY)
        return self._detect(self.front_tem_kp, self.front_tem_des, self.front_tem, input_img), self._detect(
            self.back_tem_kp,
            self.back_tem_des,
            self.back_tem,
            input_img)
