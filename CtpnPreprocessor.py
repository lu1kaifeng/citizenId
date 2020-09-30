from typing import List

import cv2
import matplotlib

from tilt_align import ctpn_coordinate_pair, get_rotation_angle, rotate_image

matplotlib.use('Agg')
from matplotlib import pyplot as plt
from ctpn.utils import image_utils, np_utils, visualize
from ctpn.utils.detector import TextDetector
from ctpn.config import cur_config as config
from ctpn.layers import models
import numpy as np

class CtpnPreProcessor:
    def __init__(self):
        config.USE_SIDE_REFINE = True
        config.IMAGES_PER_GPU = 1
        config.IMAGE_SHAPE = (1024, 1024, 3)
        self.m = models.ctpn_net(config, 'test')
        self.m.load_weights(config.WEIGHT_PATH, by_name=True)
        self.m.summary()

    def get_text_lines(self,img: np.ndarray,interactive=False)->List[ctpn_coordinate_pair]:
        # 加载图片
        image, image_meta, _, _ = image_utils.load_image_gt(np.random.randint(10),
                                                            img,
                                                            config.IMAGE_SHAPE[0],
                                                            None)

        # 加载模型

        # 模型预测
        text_boxes, text_scores, _ = self.m.predict([np.array([image]), np.array([image_meta])])
        text_boxes = np_utils.remove_pad(text_boxes[0])
        text_scores = np_utils.remove_pad(text_scores[0])[:, 0]

        # 文本行检测器
        image_meta = image_utils.parse_image_meta(image_meta)
        detector = TextDetector(config)
        text_lines = detector.detect(text_boxes, text_scores, config.IMAGE_SHAPE, image_meta['window'])
        # 可视化保存图像
        boxes_num = 30
        fig = plt.figure(figsize=(16, 16))
        ax = fig.add_subplot(1, 1, 1)
        visualize.display_polygons(image, text_lines[:boxes_num, :8], text_lines[:boxes_num, 8],
                                   ax=ax)
        lines = list(map(lambda x: ctpn_coordinate_pair(x,x[4]),text_lines))
        for r in text_boxes:
            # (y1,x1,y2,x2)
            cv2.rectangle(image, (r[1], r[0]), (r[3], r[2]), (0, 255, 0), 2)
        for r in lines:
            # (y1,x1,y2,x2)
            cv2.line(image, (r.x1, r.y1), (r.x2, r.y2), (255, 0, 0), 2)

        #image = rotate_image(image, get_rotation_angle(lines))
        if interactive:
            cv2.imshow('img', image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        return lines