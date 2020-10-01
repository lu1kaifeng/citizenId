import cv2
import numpy as np
from scipy.spatial.distance import directed_hausdorff
class FrontTextMatcher:


    def __init__(self):
        self.surf = cv2.xfeatures2d.SURF_create(500)
        self.kp, self.des = self.surf.detectAndCompute(self.img, None)
        # self.img= cv2.drawKeypoints(self.img, self.kp, None, (255, 0, 0), 4)
        # cv2.imshow('sex', self.img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

    def text_front_recog(self, front, interactive=False):
        pass
