"""互信息
    MI(x, y) = log_2(p(x,y) / p(x)*p(y))
"""
import re
import os
import math
import pickle
import numpy as np
import pandas as pd

from corpus import news_path, news_jieba_path, corpus_root_path
from tools import running_of_time

word_path = '/projects/myNlp/machine_learning/wechat_new_word_data.txt'


def read_data():
    d = {}
    idx = 0
    with open(word_path, encoding='utf8', mode='r') as f:
        for line in f.readlines():
            t = line.strip().split(',')
            d[t[0]] = int(t[1])
            idx += 1
            if idx == 1000:
                break
    return d


class MI:
    def __init__(self):
        self.data_dict = {}
        self.pre_dict = {}
        self.load_data_dict()
        self.d = None
        self.key_dict = {}

    def load_data_dict(self):
        self.data_dict = read_data()

    def mi(self, key1, key2):
        p_x_y = self.data_dict.get(key1 + key2, 1)
        p_x = self.data_dict.get(key1, 1)
        p_y = self.data_dict.get(key2, 1)
        return math.log(p_x_y / (p_x * p_y), 2)

    def create_numpy(self):
        idx = 0
        for key in self.data_dict:
            self.key_dict[key] = idx
            idx += 1
        word_len = len(self.data_dict)
        self.d = np.zeros(shape=(word_len, word_len), dtype='float32')

    def train(self):
        self.create_numpy()
        print(self.d.shape)
        for key1, value1 in self.key_dict.items():
            print(key1)
            for key2, value2 in self.key_dict.items():
                self.d[value1][value2] = self.data_dict.get(key1 + key2, 1)


class WS:
    def __init__(self):
        self.one = {}
        self.two = {}

    @running_of_time
    def train(self):
        # files = open(news_jieba_path, mode='r', encoding='gb2312').readlines()
        # for file in files:
        #     file = ''.join(file.strip().split(' ')[:-1])
        #
        #     drop_dict = ['，', '\n', '。', '、', '：', '(', ')', '[', ']', '.', ',', ' ', '\u3000', '”', '“',
        #                  '？', '?', '！', '‘', '’', '…']
        #     for i in drop_dict:  # 去掉标点字
        #         file = file.replace(i, '')
        #
        #     for idx in range(len(file) - 1):
        #         self.one[file[idx]] = self.one.get(file[idx], 0) + 1
        #         self.two[file[idx:idx + 2]] = self.two.get(file[idx:idx + 2], 0) + 1
        #     self.one[file[idx + 1]] = self.one.get(file[idx + 1], 0) + 1
        #
        # with open(os.path.join(corpus_root_path, 'keys.txt'), mode='wb') as f:
        #     pickle.dump(self.one, f)
        #     pickle.dump(self.two, f)
        # print(len(self.one))
        # print(len(self.two))

        with open(os.path.join(corpus_root_path, 'keys.txt'), mode='rb') as f:
            self.one = pickle.load(f)
            self.two = pickle.load(f)

        tmp_one = dict()
        one_sum = sum(self.one.values())
        tmp_two = dict()
        two_sum = sum(self.two.values())
        for k, v in self.one.items():
            tmp_one[k] = v / one_sum

        for k, v in self.two.items():
            tmp_two[k] = v / two_sum

        r = []
        for t in tmp_two:
            one, two = t
            tmp_two.get(t)
            score = math.log(tmp_two.get(t) / (tmp_one.get(one) * tmp_one.get(two)), 2)
            r.append((t, score))

        print(sorted(r, key=lambda x: x[1], reverse=True)[:1000])


def test():
    mi = MI()
    mi.train()


if __name__ == '__main__':
    ws = WS()
    ws.train()
