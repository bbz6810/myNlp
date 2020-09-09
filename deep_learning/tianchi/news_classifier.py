import os
import time
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from keras import models, layers
from keras.preprocessing.sequence import pad_sequences
from keras.initializers import he_normal
from corpus import news_classifier_path, news_test_path, news_one_to_one_path
from tools import running_of_time

batch_size = 32
epochs = 5
vocab_size = 10000
embedding_dim = 100
max_words = 1024
class_num = 14

# y_dict = dict()
y_dict = {'9': 0, '12': 1, '7': 2, '11': 3, '0': 4, '10': 5, '4': 6, '3': 7, '5': 8, '2': 9, '6': 10, '1': 11,
          '8': 12, '13': 13}


def pre_x(x):
    train_x = pad_sequences(x, maxlen=max_words, padding='post', truncating='post')
    return train_x


def pre_y(y):
    # set_label = set(map(str, y))
    # for i in set_label:
    #     y_dict[i] = len(y_dict)
    train_y = list()
    for b in map(str, y):
        t = np.zeros(len(y_dict))
        t[y_dict[b]] = 1
        train_y.append(t)
    train_y = np.array(train_y)
    print('pre_y, 生成y值向量', train_y.shape)
    return train_y


class FastText:
    def __init__(self):
        self.nn = None

    def model(self):
        model = models.Sequential()
        model.add(layers.Embedding(vocab_size + 1, embedding_dim, input_length=max_words))
        model.add(layers.GlobalAveragePooling1D())
        model.add(layers.Dense(class_num, activation='softmax', kernel_initializer=he_normal(5)))
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        print(model.summary())
        return model

    def train(self, train_x, train_y, test_x, test_y):
        self.nn = self.model()
        self.nn.fit(train_x, train_y, epochs=epochs, batch_size=batch_size, validation_data=(test_x, test_y))
        self.nn.save(os.path.join(news_one_to_one_path, 'text_cnn_conv2'))

    def predict(self, test_x, y_set):
        y_set = dict((v, k) for k, v in y_set.items())
        _y = self.nn.predict(test_x)
        max_arg = np.argmax(_y, axis=1)
        for i in max_arg:
            print(y_set[i])


class TextCNN:
    def __init__(self):
        self.nn = None

    def model(self):
        inputs = layers.Input(shape=(max_words,), dtype='float32')
        embedding = layers.Embedding(input_dim=vocab_size + 1, output_dim=embedding_dim, input_length=max_words)
        embed = embedding(inputs)
        # 907*100 - 907*256 -
        # cnn1 = layers.Conv1D(256, 3, padding='same', strides=1, activation='relu')(embed)
        # cnn1 = layers.MaxPool1D(pool_size=48)(cnn1)
        #
        # cnn2 = layers.Conv1D(256, 4, padding='same', strides=1, activation='relu')(embed)
        # cnn2 = layers.MaxPool1D(pool_size=47)(cnn2)
        #
        # cnn3 = layers.Conv1D(256, 5, padding='same', strides=1, activation='relu')(embed)
        # cnn3 = layers.MaxPool1D(pool_size=46)(cnn3)

        cnn1 = layers.Conv1D(256, 3, padding='same', strides=1, activation='relu')(embed)
        cnn1 = layers.MaxPool1D(pool_size=32)(cnn1)
        # cnn1 = layers.Conv1D(512, 3, padding='same', strides=1, activation='relu')(cnn1)
        # cnn1 = layers.MaxPool1D(pool_size=8)(cnn1)

        cnn2 = layers.Conv1D(256, 4, padding='same', strides=1, activation='relu')(embed)
        cnn2 = layers.MaxPool1D(pool_size=31)(cnn2)
        # cnn2 = layers.Conv1D(512, 4, padding='same', strides=1, activation='relu')(cnn2)
        # cnn2 = layers.MaxPool1D(pool_size=7)(cnn2)

        cnn3 = layers.Conv1D(256, 5, padding='same', strides=1, activation='relu')(embed)
        cnn3 = layers.MaxPool1D(pool_size=30)(cnn3)
        # cnn3 = layers.Conv1D(512, 6, padding='same', strides=1, activation='relu')(cnn3)
        # cnn3 = layers.MaxPool1D(pool_size=6)(cnn3)

        cnn_con = layers.concatenate(inputs=[cnn1, cnn2, cnn3], axis=1)

        cnn4 = layers.Conv1D(512, 3, padding='same', strides=1, activation='relu')(cnn_con)
        cnn4 = layers.MaxPool1D(pool_size=8)(cnn4)
        cnn5 = layers.Conv1D(512, 4, padding='same', strides=1, activation='relu')(cnn_con)
        cnn5 = layers.MaxPool1D(pool_size=7)(cnn5)
        cnn6 = layers.Conv1D(512, 6, padding='same', strides=1, activation='relu')(cnn_con)
        cnn6 = layers.MaxPool1D(pool_size=6)(cnn6)

        cnn_con2 = layers.concatenate(inputs=[cnn4, cnn5, cnn6], axis=1)

        flat = layers.Flatten()(cnn_con2)
        drop = layers.Dropout(0.5)(flat)
        output = layers.Dense(class_num, activation='sigmoid')(drop)
        model = models.Model(inputs=inputs, outputs=output)
        print(model.summary())
        return model

    @running_of_time
    def train(self, train_x, train_y, test_x, test_y):
        self.nn = self.model()
        # self.nn = self.create_model()
        self.nn.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        self.nn.fit(train_x, train_y, epochs=epochs, batch_size=batch_size, validation_data=(test_x, test_y))
        self.nn.save(os.path.join(news_one_to_one_path, 'text_cnn_conv2'))

    def predict(self, test_x, y_set):
        y_set = dict((v, k) for k, v in y_set.items())
        _y = self.nn.predict(test_x)
        max_arg = np.argmax(_y, axis=1)
        for i in max_arg:
            print(y_set[i])

    def create_model(self):
        inputs = layers.Input(shape=(max_words,))
        embedding = layers.Embedding(input_dim=vocab_size + 1, output_dim=embedding_dim, input_length=max_words)(inputs)
        reshape = layers.Reshape((max_words, embedding_dim, 1))(embedding)

        conv_0 = layers.Conv2D(256, (3, embedding_dim), padding='valid', activation='relu')(reshape)
        maxpool_0 = layers.MaxPool2D((max_words - 3 + 1, 1), strides=(1, 1), padding='valid')(conv_0)

        conv_1 = layers.Conv2D(256, (4, embedding_dim), padding='valid', activation='relu')(reshape)
        maxpool_1 = layers.MaxPool2D((max_words - 4 + 1, 1), strides=(1, 1), padding='valid')(conv_1)

        conv_2 = layers.Conv2D(256, (5, embedding_dim), padding='valid', activation='relu')(reshape)
        maxpool_2 = layers.MaxPool2D((max_words - 5 + 1, 1), strides=(1, 1), padding='valid')(conv_2)

        flat = layers.concatenate(inputs=[maxpool_0, maxpool_1, maxpool_2], axis=1)

        flatten = layers.Flatten()(flat)
        drop = layers.Dropout(0.4)(flatten)

        output = layers.Dense(class_num, activation='softmax')(drop)

        model = models.Model(inputs=inputs, output=output)
        print(model.summary())
        return model

    def load_model(self):
        self.nn = models.load_model(os.path.join(news_one_to_one_path, 'text_cnn_conv2'))


@running_of_time
def format_data(data_path):
    x, y = [], []
    with open(data_path, mode='r') as f:
        for line in f.readlines()[1:]:
            line = line.strip().split('\t')
            x.append(list(map(int, line[1].split())))
            y.append(int(line[0]))
    return x, y


@running_of_time
def format_test_data(data_path):
    x = []
    data = pd.read_csv(data_path, sep='\t')
    for i in data.values:
        x.append(list(map(int, i[0].split())))
    return x


def set_vocab_size(x):
    global vocab_size
    vocab_size = max(max(i) for i in x)
    print('vocab_size', vocab_size)
    max_len = max(len(i) for i in x)
    print('pre_x, 最大长度', max_len)
    avg_len = sum(len(i) for i in x) / len(x)
    global max_words
    max_words = int(avg_len)
    print('pre_x, 平均长度', avg_len)
    """
    vocab_size 7549
    pre_x, 最大长度 57921
    pre_x, 平均长度 907.20711

    """


def train():
    x, y = format_data(news_classifier_path)
    # set_vocab_size(x)
    train_x, train_y = pre_x(x), pre_y(y)
    location = np.random.permutation(len(train_x))
    train_x = train_x[location]
    train_y = train_y[location]

    train_x, test_x, train_y, test_y = train_test_split(train_x, train_y, train_size=0.1)
    _, test_x, _, test_y = train_test_split(test_x, test_y, train_size=0.9)
    print('label', y_dict)
    # textcnn = FastText()
    textcnn = TextCNN()
    textcnn.train(train_x, train_y, test_x, test_y)
    # loss, acc = textcnn.nn.evaluate(test_x, test_y)
    # print('acc', acc, 'loss', loss)


def predict():
    textcnn = TextCNN()
    textcnn.load_model()

    test_x = format_test_data(news_test_path)
    test_x = pre_x(test_x)
    textcnn.predict(test_x, y_dict)


def predict2():
    textcnn = TextCNN()
    textcnn.load_model()
    x, y = format_data(news_classifier_path)
    set_vocab_size(x)
    train_x, train_y = pre_x(x), pre_y(y)
    error = []
    for i in range(100):
        location = np.random.permutation(len(train_x))
        train_x = train_x[location][:10000]
        train_y = train_y[location][:10000]

        loss, acc = textcnn.nn.evaluate(train_x, train_y)
        print(i, 'acc', acc, 'loss', loss)
        error.append((acc, loss))
    print('accuracy')
    print(sum([i[0] for i in error]))
    print(sum([i[0] for i in error]) / len(error))
    print('loss')
    print(sum([i[1] for i in error]))
    print(sum([i[1] for i in error]) / len(error))


if __name__ == '__main__':
    train()
    # predict2()
