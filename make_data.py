import re
from os import listdir
from os.path import isfile, join

import pandas
import json
import numpy as np

read = open('./citizen_id_words.txt', 'r')
all_label_dict = read.read()
all_label_dict = eval(all_label_dict)


def get_unique_words():
    csv = pandas.read_csv(r'D:\citizenIdData\Train_Labels.csv', header=None).values
    txt = []
    for c in csv:
        c[4] = str(c[4]) + '年' + str(c[5]) + '月' + str(c[6]) + '日'
        txt.append(np.delete(np.delete(c, 5), 5))
    my_dict = {}
    for c in txt:
        my_dict[c[0]] = {
            'pd': c[7],
            'period': c[8],
            'id': c[6],
            'address': c[5],
            'birth': c[4],
            'ethnic': c[2],
            'name': c[1],
            'sex': c[3]
        }
    return my_dict


def words_list2label_list(words_list):
    """
    将图像中单个文字label拼成label_list
    :param words:
    :return:
    """
    label_list = []
    for words in words_list:
        for i in words:
            if i == ' ' or i == '　':
                continue
            if i in all_label_dict.keys():
                label_list.append(all_label_dict[i])
            else:
                print(i)

    return label_list


if __name__ == '__main__':
    word_dict = get_unique_words()
    path = r'../../data/ctc_train'
    label_dict = {}
    files = [f for f in listdir(path) if isfile(join(path, f)) and re.match('.*\\.jpg', f)]
    for file in files:
        f = re.match('(.*)\\.jpg-(.*)\\.jpg', file).group(1)
        t = re.match('(.*)\\.jpg-(.*)\\.jpg', file).group(2)
        label_dict[r'./data/ctc_train/' + file] = word_dict[f][t]
    for k, v in label_dict.items():
        label_dict[k] = words_list2label_list(v)
    print(label_dict, file=open('citizen_id_label.txt', 'w'))
