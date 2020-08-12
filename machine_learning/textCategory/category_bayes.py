import os

import jieba

from tools import filter_stop, fetch_file_path
from model.bayes import Bayes
from corpus import category_path


class Category:
    def __init__(self):
        self.classifier = Bayes()

    def save(self):
        self.classifier.save('20200710')

    def load(self):
        self.classifier.load('20200710')

    def train(self):
        data = []
        for key, path in category_path.items():
            for txt_path in fetch_file_path(path):
                current_path = os.path.join(path, txt_path)
                with open(current_path, mode='r', encoding='gb2312', errors='ignore') as f:
                    data.append(
                        [filter_stop(jieba.cut(''.join((map(lambda x: x.strip().replace(' ', ''), f.readlines()))))),
                         key])
        self.classifier.train(data=data)

    def predict(self, sent):
        words = filter_stop(jieba.cut(sent))
        return self.classifier.predict(words)


if __name__ == '__main__':
    c = Category()
    # c.train()
    # c.save()
    c.load()
    d = c.predict('在生活中，技能是一个非常重要的能力，如果有人会则有很大提升')
    print(d)
