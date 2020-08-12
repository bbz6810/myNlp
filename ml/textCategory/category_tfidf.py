import os

import jieba

from tools import filter_stop, fetch_file_path
from model.tf_idf import TFIDF
from model.feature_select.gain import Gain
from corpus import category_path


class Category:
    def __init__(self):
        self.classifier = TFIDF()

    def save(self):
        self.classifier.save(os.path.join(os.path.dirname(os.path.abspath(__file__)), '20200711'))

    def load(self):
        self.classifier.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), '20200711'))

    def train(self):
        data = []
        for key, path in category_path.items():
            for txt_path in fetch_file_path(path):
                current_path = os.path.join(path, txt_path)
                with open(current_path, mode='r', encoding='gb2312', errors='ignore') as f:
                    data.append(
                        [filter_stop(jieba.cut(''.join((map(lambda x: x.strip().replace(' ', ''), f.readlines()))))),
                         key])
        self.classifier.train(data)

    def gain(self):
        self.load()
        data = [[a, b] for a, b in zip(self.classifier.tf_idf, self.classifier.labels)]
        gain = Gain()
        used_attr = []
        for i in range(100):
            t = gain.calc_gain_all(data, used_attr)
            used_attr.append(t)
            d = {v: k for k, v in self.classifier.word_index.items()}
            print(i, t, d[t])


if __name__ == '__main__':
    c = Category()
    # c.train()
    # c.save()
    # c.load()
    # print(c.classifier.tf_list[:1])
    # print(c.classifier.idf.values())
    # c.train()
    # c.save()
    # c.load()
    c.gain()
    # print(d)
