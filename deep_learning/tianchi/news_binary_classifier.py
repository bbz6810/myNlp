import numpy as np
import pandas as pd
import os

from keras import models, layers
from keras.preprocessing.sequence import pad_sequences
from keras.initializers import he_normal
from corpus import news_classifier_path, news_test_path, news_one_to_one_path

batch_size = 32
epochs = 16
vocab_size = 10000
embedding_dim = 100
max_words = 1000
class_num = 1

y_dict = dict()

"""build many one to one classifier or many one to many classifier

"""


def every_class_data_collect():
    data_dict = dict()
    data = pd.read_csv(news_classifier_path, sep='\t')
    for i in data.values:
        if str(i[0]) not in data_dict:
            data_dict[str(i[0])] = list()
        data_dict[str(i[0])].append(list(map(int, i[1].split())))
    return data_dict


def fetch_one2one(one, two, datas):
    x = datas[one] + datas[two]
    y = [0] * len(datas[one]) + [1] * len(datas[two])
    return x, y


def pre_x(x):
    train_x = pad_sequences(x, maxlen=max_words, padding='post', truncating='post')
    return train_x


def set_vocab_size(x):
    global vocab_size
    vocab_size = max(max(i) for i in x)
    print('vocab_size', vocab_size)
    max_len = max(len(i) for i in x)
    print('pre_x, 最大长度', max_len)
    avg_len = sum(len(i) for i in x) / len(x)
    # global max_words
    # max_words = avg_len
    print('pre_x, 平均长度', avg_len)
    """
    vocab_size 7549
    pre_x, 最大长度 57921
    pre_x, 平均长度 907.20711

    """


def train(datas):
    for one in datas:
        for two in datas:
            if one == two:
                continue
            x, y = fetch_one2one(one, two, datas)
            x = pre_x(x)
            set_vocab_size(x)
            fasttext = FastText()
            fasttext.train(x, y)
            fasttext.save(os.path.join(news_one_to_one_path, '{}_{}'.format(one, two)))
            # return


class FastText:
    def __init__(self):
        self.nn = None

    def model(self):
        model = models.Sequential()
        model.add(layers.Embedding(vocab_size + 1, embedding_dim, input_length=max_words))
        model.add(layers.GlobalAveragePooling1D())
        model.add(
            layers.Dense(class_num, activation='sigmoid', kernel_initializer=he_normal(3), kernel_regularizer='l2'))
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        print(model.summary())
        return model

    def train(self, train_x, train_y):
        self.nn = self.model()
        self.nn.fit(train_x, train_y, epochs=epochs, batch_size=batch_size, validation_split=0.4)

    def save(self, path):
        self.nn.save(path)

    def predict(self, test_x):
        _y = self.nn.predict(test_x)


if __name__ == '__main__':
    data_dict = every_class_data_collect()
    train(data_dict)
