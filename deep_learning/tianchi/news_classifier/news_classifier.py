import os
import sys
import pickle

sys.path.append('/Users/zhoubb/projects/myNlp')

import numpy as np
from sklearn.model_selection import train_test_split, KFold
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical

from corpus import tianchi_news_class_path, news_classifier_path, news_test_path
from tools import running_of_time
from deep_learning.tianchi.news_classifier.base_model import TextRNN, FastText, TextCNN


class TCNewsClass:
    def __init__(self, cls, embedding_dim=64, epochs=8, batch_size=32, samples=10000):
        self.vocab_size = 0
        self.word_vocab = dict()
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
            'word_vocab': self.word_vocab,
            'embedding_dim': self.embedding_dim,
            'max_words': self.max_words,
            'class_num': self.class_num,
            'epochs': self.epochs,
            'batch_size': self.batch_size
        }
        # print(config)
        return config

    def set_config(self, config):
        self.__dict__.update(config)

    def padding_x(self, x):
        self.set_param(x)
        train_x = pad_sequences(x, maxlen=self.max_words, padding='post', truncating='post', value=self.vocab_size - 1)
        print('train x shape', train_x.shape)
        return train_x

    def padding_y(self, y):
        self.class_num = len(set(y))
        return to_categorical(y)

    def padding_test_x(self, x):
        train_x = pad_sequences(x, maxlen=self.max_words, padding='post', truncating='post', value=self.vocab_size - 1)
        print('test x shape', train_x.shape)
        return train_x

    @running_of_time
    def format_train_data(self, data_path):
        x, y = [], []
        with open(data_path, mode='r') as f:
            for line in f.readlines()[1:]:
                line = line.strip().split('\t')
                x.append(list(map(int, line[1].split())))
                y.append(int(line[0]))
        return x, y

    @running_of_time
    def format_test_data(self, data_path):
        x = []
        with open(data_path, mode='r') as f:
            for line in f.readlines()[1:]:
                line = line.strip().split('\t')
                x.append(list(map(int, line[1].split())))
        return x

    @running_of_time
    def format_test_data(self, data_path):
        x = []
        with open(data_path, mode='r') as f:
            for line in f.readlines()[1:]:
                x.append(list(map(int, line.strip().split())))
        return x

    @running_of_time
    def set_param(self, x):
        self.vocab_size = max(self.word_vocab) + 10
        # self.vocab_size = 7600
        print('vocab size', self.vocab_size)
        ml = max(len(i) for i in x)
        print('文本最大长度', ml)
        # self.max_words = int(sum(len(i) for i in x) / len(x))
        self.max_words = 2400
        print('设置文本最大长度为平均长度', self.max_words)

    def shuffle(self, x, y):
        np.random.seed(1)
        location = np.random.permutation(len(x))
        # return x[location][:20000], y[location][:20000]
        return x[location], y[location]

    def train(self):
        # vocab size 7559
        # 文本最大长度 57921
        # 设置文本最大长度为平均长度 2400
        # x, y = self.format_train_data(news_classifier_path)
        # test_x = self.format_test_data(news_test_path)
        # # 增加词字典
        # self.word_vocab.update(dict((k, k) for _x in x for k in _x))
        # self.word_vocab.update(dict((k, k) for _x in test_x for k in _x))
        # x, y = self.padding_x(x), self.padding_y(y)
        # self.save_data(x, y)
        # return
        x, y = self.load_data()
        x, y = self.shuffle(x, y)
        print('x shape', x.shape, 'y shape', y.shape)
        config = self.get_config()
        config['embedding_dim'] = 400
        # config['batch_size'] = 64

        # train_x, test_x, train_y, test_y = train_test_split(x, y, train_size=0.8, shuffle=True)
        # ft = self.cls(**config)
        # ft.train(train_x, train_y)
        # ft.predict(test_x, test_y)

        print('start k fold')
        kfold = KFold(n_splits=5, shuffle=True, random_state=2020)
        for i, (train_index, test_index) in enumerate(kfold.split(x)):
            train_x, train_y = x[train_index], y[train_index]
            test_x, test_y = x[test_index], y[test_index]
            ft = self.cls(**config)
            ft.train(train_x, train_y, test_x, test_y)

    def predict(self):
        _, _ = self.load_data()
        x = self.format_test_data(news_test_path)
        x = self.padding_test_x(x)
        config = self.get_config()
        nn = self.cls(**config)
        nn.load_model()
        nn.evaluate(x)

    def save_data(self, x, y):
        with open(os.path.join(tianchi_news_class_path, 'train_data_2400.pkl'), mode='wb') as f:
            pickle.dump(x, f)
            pickle.dump(y, f)
            pickle.dump(self.get_config(), f)

    @running_of_time
    def load_data(self):
        with open(os.path.join(tianchi_news_class_path, 'train_data_2400.pkl'), mode='rb') as f:
            x = pickle.load(f)
            y = pickle.load(f)
            config = pickle.load(f)
        self.set_config(config)
        return x, y


def running():
    tc = TCNewsClass(cls=TextCNN, embedding_dim=400, epochs=5)
    tc.train()
    # tc.predict()


if __name__ == '__main__':
    running()
