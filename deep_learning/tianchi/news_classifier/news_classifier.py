import os
import sys
import pickle

sys.path.append('/Users/zhoubb/projects/myNlp')

import numpy as np
from sklearn.model_selection import train_test_split
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical

from corpus import tianchi_news_class_path
from tools import running_of_time
from deep_learning.tianchi.news_classifier.base_model import TextRNN, FastText


class TCNewsClass:
    def __init__(self, cls, embedding_dim=64, epochs=8, batch_size=32, samples=10000):
        self.vocab_size = 0
        self.max_words = 0
        self.class_num = 0
        self.embedding_dim = embedding_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.samples = samples
        self.cls = cls

    def get_config(self):
        config = {
            'vocab_size': self.vocab_size,
            'embedding_dim': self.embedding_dim,
            'max_words': self.max_words,
            'class_num': self.class_num,
            'epochs': self.epochs,
            'batch_size': self.batch_size
        }
        print(config)
        return config

    def set_config(self, config):
        self.__dict__.update(config)

    def padding_x(self, x):
        self.set_param(x)
        train_x = pad_sequences(x, maxlen=self.max_words, padding='post', truncating='post', value=7600)
        return train_x

    def padding_y(self, y):
        self.class_num = len(set(y))
        return to_categorical(y)

    @running_of_time
    def format_train_data(self, data_path):
        x, y = [], []
        with open(data_path, mode='r') as f:
            for line in f.readlines()[1:self.samples]:
                line = line.strip().split('\t')
                x.append(list(map(int, line[1].split())))
                y.append(int(line[0]))
        return x, y

    @running_of_time
    def format_test_data(self, data_path):
        x = []
        with open(data_path, mode='r') as f:
            for line in f.readlines()[1:]:
                x.append(list(map(int, line.strip().split())))
        return x

    @running_of_time
    def set_param(self, x):
        # self.vocab_size = max(max(i) for i in x)
        self.vocab_size = 7600
        print('vocab size', self.vocab_size)
        ml = max(len(i) for i in x)
        print('文本最大长度', ml)
        # self.max_words = int(sum(len(i) for i in x) / len(x))
        self.max_words = 2000
        print('设置文本最大长度为平均长度', self.max_words)

    def shuffle(self, x, y):
        np.random.seed(1)
        location = np.random.permutation(len(x))
        return x[location][:20000], y[location][:20000]
        # return x[location], y[location]

    def train(self):
        # x, y = self.format_train_data(news_classifier_path)
        # x, y = self.padding_x(x), self.padding_y(y)
        # self.save_data(x, y)
        x, y = self.load_data()
        x, y = self.shuffle(x, y)
        print('x shape', x.shape, 'y shape', y.shape)
        train_x, test_x, train_y, test_y = train_test_split(x, y, train_size=0.8, shuffle=True)
        config = self.get_config()
        ft = self.cls(**config)
        ft.train(train_x, train_y)
        ft.predict(test_x, test_y)

    def save_data(self, x, y):
        with open(os.path.join(tianchi_news_class_path, 'train_data_2000.pkl'), mode='wb') as f:
            pickle.dump(x, f)
            pickle.dump(y, f)
            pickle.dump(self.get_config(), f)

    @running_of_time
    def load_data(self):
        with open(os.path.join(tianchi_news_class_path, 'train_data.pkl'), mode='rb') as f:
            x = pickle.load(f)
            y = pickle.load(f)
            config = pickle.load(f)
        self.set_config(config)
        return x, y


def running():
    tc = TCNewsClass(cls=FastText, embedding_dim=128, epochs=5, samples=100000)
    tc.train()


if __name__ == '__main__':
    running()
