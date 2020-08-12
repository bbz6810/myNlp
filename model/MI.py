"""互信息
    MI(x, y) = log_2(p(x,y) / p(x)*p(y))
"""

import math
import numpy as np

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


def test():
    mi = MI()
    mi.train()


if __name__ == '__main__':
    test()
