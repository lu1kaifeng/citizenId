import re
from os import listdir
from os.path import isfile, join

import cv2
import matplotlib
import numpy as np
import random as rng
import matplotlib.pyplot as plt
from rect_op import merge_bounding_boxes

matplotlib.use('Agg')
from matplotlib import pyplot as plt
from ctpn.utils import image_utils, np_utils, visualize
from ctpn.utils.detector import TextDetector
from ctpn.config import cur_config as config
from ctpn.layers import models

config.USE_SIDE_REFINE = True
config.IMAGES_PER_GPU = 1
config.IMAGE_SHAPE = (1024, 1024, 3)
m = models.ctpn_net(config, 'test')
m.load_weights(config.WEIGHT_PATH, by_name=True)
m.summary()

def is_correct_crop(name: str, dir=r'D:\citizenIdData\train'):

    # 加载图片
    image, image_meta, _, _ = image_utils.load_image_gt(np.random.randint(10),
                                                        dir + '\\' + name,
                                                        config.IMAGE_SHAPE[0],
                                                        None)

    # 加载模型

    # 模型预测
    text_boxes, text_scores, _ = m.predict([np.array([image]), np.array([image_meta])])
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
    for r in text_boxes:
        cv2.rectangle(image, (r[1], r[0]), (r[3] , r[2] ), (0, 255, 0), 2)
    cv2.imshow('img',image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


path = r'D:\citizenIdData\Train_DataSet'
files = [f for f in listdir(path) if isfile(join(path, f)) and re.match('.*\\.jpg', f)]

for file in files:
    is_correct_crop(file + '-0.jpg')
    is_correct_crop(file + '-1.jpg')
