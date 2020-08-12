import os

import jieba

from model.bayes import Bayes
from tools import filter_stop

pos_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        'pos.txt')
neg_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        'neg.txt')
stop_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         'stopwords.txt')


class Emotion:
    def __init__(self):
        self.classifier = Bayes()

    def save(self):
        self.classifier.save('20200710')

    def load(self):
        self.classifier.load('20200710')

    def train(self):
        data = []
        with open(pos_path, encoding='utf8', mode='r') as f:
            for line in f.readlines():
                data.append([filter_stop(jieba.cut(line.strip())), 'pos'])
        with open(neg_path, encoding='utf8', mode='r') as f:
            for line in f.readlines():
                data.append([filter_stop(jieba.cut(line.strip())), 'neg'])
        self.classifier.train(data=data)

    def classify(self, sent):
        filter_words = filter_stop(jieba.cut(sent.strip()))
        return self.classifier.predict(filter_words)


if __name__ == '__main__':
    c = '千万别买，我很后悔'
    b = Emotion()
    # b.train()
    # b.save()
    b.load()
    res = b.classify(c)
    print(res)
