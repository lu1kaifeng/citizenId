import numpy as np
import cv2


def get_text_img(front: np.ndarray, back: np.ndarray):
    dict = {
        "pd": back[ 208:242,98:338],
        "period": back[ 245:280,103:360],
        "name": front[ 38:74,71:153],
        "sex": front[ 76:109,76:113],
        "ethnic": front[ 74:107,176:245],
        "birth": front[ 112:146,74:243],
        "address": front[ 146:201,78:300],
        "id": front[ 238:281,123:385]
    }
    for k, v in dict.items():
        cv2.imshow(k, v)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
