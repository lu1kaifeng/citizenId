import numpy as np
import pandas as pd
import tensorflow as tf
import cv2
from cnn_rnn_ctc.crnn import CRNN
from cnn_rnn_ctc.utils.net_cfg_parser import parser_cfg_file

class Test_CRNN(object):
    def __init__(self, batch_size=None,interactive=False):
        self.interactive = interactive
        net_params, train_params = parser_cfg_file('./net.cfg')
        self._model_save_path = str(net_params['model_load_path'])
        self.input_img_height = int(net_params['input_height'])
        self.input_img_width = int(net_params['input_width'])
        if batch_size is None:
            self.test_batch_size = int(net_params['test_batch_size'])
        else:
            self.test_batch_size = batch_size

        # 加载label onehot
        f = open('./data/citizen_id_words.txt', 'r')
        data = f.read()
        words_onehot_dict = eval(data)
        self.words_list = list(words_onehot_dict.keys())
        self.words_onehot_list = [words_onehot_dict[self.words_list[i]] for i in range(len(self.words_list))]

        # 构建网络
        self.inputs_tensor = tf.placeholder(tf.float32, [self.test_batch_size, self.input_img_height, self.input_img_width, 1])
        self.seq_len_tensor = tf.placeholder(tf.int32, [None], name='seq_len')

        crnn_net = CRNN(net_params, self.inputs_tensor, self.seq_len_tensor, self.test_batch_size, True)
        net_output, decoded, self.max_char_count = crnn_net.construct_graph()
        self.dense_decoded = tf.sparse_tensor_to_dense(decoded[0], default_value=-1)

        self.sess = tf.Session()
        saver = tf.train.Saver()
        saver.restore(self.sess, self._model_save_path)

    def _get_input(self, img):


        batch_data = np.zeros([1,
                               self.input_img_height,
                               self.input_img_width,
                               1])
        img_list = [img]

        # print(np.shape(img))
        # print(img_path_list[i])
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        resized_img = self._resize_img(img)
        reshape_img = resized_img.reshape([1, self.input_img_height, self.input_img_width, 1])
        img_norm = reshape_img / 255 * 2 - 1
        batch_data[0] = img_norm

        return batch_data, 1, img_list


    def test_img(self, img, is_show_res=False):

        batch_data, batch_size, img_list= self._get_input(img)
        if batch_size != self.test_batch_size:
            error = '网络构建batch size:'+str(self.test_batch_size)+'和实际输入batch size:'+str(batch_size)+'不一样'
            assert 0, error

        feed_dict = {self.inputs_tensor: batch_data, self.seq_len_tensor: [self.max_char_count]*batch_size}
        predict = self.sess.run(self.dense_decoded, feed_dict=feed_dict)
        predict_seq = self._predict_to_words(predict)

        if is_show_res:
            for i in range(batch_size):
                cv2.imshow(img, img_list[i])
            cv2.waitKey()

        return predict_seq

    def _predict_to_words(self, decoded):
        words = []

        for seq in decoded:
            seq_words = ''
            for onehot in seq:
                if onehot == -1:
                    break
                seq_words += self.words_list[self.words_onehot_list.index(onehot)]
            words.append(seq_words)
        return words

    def get_text_img(self,front: np.ndarray, back: np.ndarray)->dict:
        def arrange_lines(img: np.ndarray):
            height = img.shape[0]
            line_height = int(height / 3)
            return np.concatenate(
                (img[0:line_height, :], img[line_height:line_height * 2, :], img[line_height * 2:line_height * 3, :]),
                axis=1)
        id_dict = {
            "pd": back[204:240, 169:388],
            "period": back[245:280, 168:363],
            "name": front[48:84, 71:153],
            "sex": front[86:119, 76:113],
            "ethnic": front[84:117, 176:245],
            "birth": front[112:146, 74:243],
            "address": arrange_lines(front[151:225, 78:300]),
            "id": front[230:272, 123:385]
        }
        if self.interactive:
            for k, v in id_dict.items():
                cv2.imshow(k,v)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        for k, v in id_dict.items():
            id_dict[k] = self.test_img(v)
        return id_dict


    def _resize_img(self, img):
        """
        将图像先转为灰度图，并将图像进行resize
        :param img:
        :return:
        """
        if len(np.shape(img)) is 3:
            img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        height, width = np.shape(img)

        if width > self.input_img_width:
            width = self.input_img_width
            ratio = float(self.input_img_width) / width
            outout_img = cv2.resize(img, (self.input_img_width,self.input_img_height))
        else:
            outout_img = np.zeros([self.input_img_height, self.input_img_width])
            ratio = self.input_img_height / height
            img_resized = cv2.resize(img, (int(width * ratio), self.input_img_height))
            outout_img[:, 0:np.shape(img_resized)[1]] = img_resized

        return outout_img