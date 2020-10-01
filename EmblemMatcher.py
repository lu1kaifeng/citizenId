import cv2
import numpy as np

from tilt_align import rotate_image


class EmblemMatcher:
    emblem_img = cv2.imread('./data/emblem.png')

    def __init__(self):
        self.surf = cv2.xfeatures2d.SURF_create(500)
        self.kp, self.des = self.surf.detectAndCompute(self.emblem_img, None)

    def flann_match(self,img,interactive=False):
        kp2, des2 = self.surf.detectAndCompute(img, None)
        # FLANN parameters
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)  # or pass empty dictionary

        flann = cv2.FlannBasedMatcher(index_params, search_params)

        matches = flann.knnMatch(self.des, des2, k=2)
        good_matches = []
        # Need to draw only good matches, so create a mask
        matches_mask = [[0, 0] for i in range(len(matches))]

        # ratio test as per Lowe's paper
        for i, (m, n) in enumerate(matches):
            if m.distance < 0.7 * n.distance:
                matches_mask[i] = [1, 0]
                good_matches.append(m)

        draw_params = dict(matchColor=(0, 255, 0),
                           singlePointColor=(255, 0, 0),
                           matchesMask=matches_mask,
                           flags=0)
        if interactive:
            img3 = cv2.drawMatchesKnn(self.emblem_img, self.kp, img, kp2, matches, None, **draw_params)

            cv2.imshow('match',img3)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        return good_matches,kp2

    def side_orientation_recog(self,p0,p0_lines,p1,p1_lines,interactive=False):
        p0_rows, p0_cols,_ = p0.shape
        p1_rows, p1_cols,_ = p1.shape
        p0_matches,p0_kp = self.flann_match(p0,interactive=interactive)
        p1_matches,p1_kp = self.flann_match(p1,interactive=interactive)
        if len(p0_matches) >= len(p1_matches):
            back,back_kp,back_matches,back_row,back_col,back_lines = p0,p0_kp,p0_matches,p0_rows,p0_cols,p0_lines
            front,front_kp,front_matches,front_row,front_col,front_lines = p1,p1_kp,p1_matches,p1_rows,p1_cols,p1_lines
        else:
            front,front_kp,front_matches,front_row,front_col,front_lines  = p0,p0_kp,p0_matches,p0_rows,p0_cols,p0_lines
            back,back_kp,back_matches,back_row,back_col,back_lines= p1,p1_kp,p1_matches,p1_rows,p1_cols,p1_lines
        # For each match...
        points = []
        for mat in back_matches:
            # Get the matching keypoints for each of the images
            img2_idx = mat.trainIdx

            # x - columns
            # y - rows
            # Get the coordinates
            (x2, y2) = back_kp[img2_idx].pt

            points.append((x2, y2))
        flip_back = False
        for p in points:
            if not (0 <= p[0] <= int(back_col / 2)):
                flip_back = True
                break
        return flip_back,front,front_lines,back,back_lines
