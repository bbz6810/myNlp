"""
    网页链接：https://blog.csdn.net/a819825294/article/details/52438625
    实现：https://blog.csdn.net/qq_34290470/article/details/102843763

    1、输入层：词的one-hot表示，2c个向量
    2、投影层：2c个向量做累加
    3、输出层：softmax做预测，以语料中出现过的词作为叶子节点，以各词在语料中出现的次数当权值

    sigmoid 函数 1 / (1 + e^-x)
        = (1 + e^-x)^-1
        = -1 * (1 + e^-x)^-2 * (1 + e^-x)'
        = -1 * (1 + e^-x)^-2 * e^-x * -1
        = (1 + e^-x)^-2 * e^-x
        = (1+e^-x) / (1+e^-x)^-2 - 1 / (1+e^-x)^2
        = 1 / (1+e^-x) - 1 / (1+e^-x)^2
        = 1 / (1+e^-x) * (1 - 1 / (1+e^-x))
        = sigmoid * (1 - sigmoid)

        sigmoid' = sigmoid * (1 - sigmoid)
        (1 - sigmoid)' = - sigmoid * (1 - sigmoid)

        log(sigmoid)' = (1 / sigmoid) * (sigmoid * (1 - sigmoid))
                      = 1 - sigmoid

        log(1-sigmoid)' = (1 / (1-sigmoid) * (-sigmoid) * (1-sigmoid))
                        = -sigmoid

    1、输入向量为w的上下c个词，共2c个词
    2、映射层把输入层的词做加和
    3、构造层次化softmax哈夫曼树

    定义变量表示：
        1、pw代表w词的路径
        2、lw代表w词路径的长度
        3、p_w_i路径w上
"""

import os

import numpy as np
import jieba
from keras import models, layers
from gensim.models import word2vec

from tools import fetch_file_path
from corpus import category_path

vocab_size = 2000
embedding_dim = 128
max_words = 500
class_num = 2

from model.data_structure.huffman import Huffman
from tools.load import load_paper_data


def gen_word_tf_index(data_list):
    word_tf = dict()
    word_index = dict()
    for row in data_list:
        for key in row:
            word_tf[key] = word_tf.get(key, 0) + 1
            if key not in word_index:
                word_index[key] = len(word_index)
    word_index['b'] = len(word_index)
    word_index['e'] = len(word_index)
    return word_tf, word_index


class WordVec:
    def __init__(self):
        self.huffman = Huffman()
        self.word_tf = None
        self.word_index = None

    def transform_data(self, data_list):
        return [['b', 'b'] + data + ['e', 'e'] for data in data_list]

    def create_numpy(self, index, data):
        words = np.zeros(shape=(len(self.word_index)), dtype='int8')
        for i in [index - 2, index - 1, index + 1, index + 2]:
            t = np.zeros(shape=(len(self.word_index)), dtype='int8')
            t[self.word_index[data[i]]] = 1
            words += t

        one = np.zeros(shape=(len(self.word_index)), dtype='int8')
        one[self.word_index[data[index]]] = 1

        return words, one

    def train(self):
        data_list = load_paper_data()
        self.word_tf, self.word_index = gen_word_tf_index(data_list)
        transform_data_list = self.transform_data(data_list)
        # print(transform_data_list)
        for data in transform_data_list:
            print(data)
            for i in range(2, len(data) - 2):
                print(data[i - 2:i + 3], data[i])
                words, one = self.create_numpy(i, data)
                print(words)
                print(words.shape)
                print(one)
                print(one.shape)
                return
            print()

            # node_list = self.huffman.create_huffman_node_list(word_tf)
            # self.huffman.build_tree(node_list[:10])

    def predict(self):
        pass


def test():
    data = []
    for key, path in category_path.items():
        for txt_path in fetch_file_path(path):
            current_path = os.path.join(path, txt_path)
            with open(current_path, mode='r', encoding='gb2312', errors='ignore') as f:
                # print(f.readlines()[0])
                data.append(jieba.lcut(''.join((map(lambda x: x.strip().replace(' ', ''), f.readlines())))))

    word2vec_model = word2vec.Word2Vec(data, size=embedding_dim, window=5, min_count=3, workers=4)
    # print(word2vec_model)
    # print(dir(word2vec_model))
    # print(word2vec_model.wv.vocab)

    raw_input = [item for sublist in data for item in sublist]
    # print(raw_input)

    text_stream = [word for word in raw_input if word in word2vec_model.wv.vocab]
    # print(text_stream)

    seq_length = 10
    x, y = [], []
    for i in range(len(text_stream) - seq_length):
        given = text_stream[i:i + seq_length]
        predict = text_stream[i + seq_length]
        x.append(np.array([word2vec_model[w] for w in given]))
        y.append(word2vec_model[predict])

    x = np.reshape(x, (-1, seq_length, embedding_dim))
    y = np.reshape(y, (-1, embedding_dim))

    model = models.Sequential()
    model.add(layers.LSTM(256, dropout_W=0.2, dropout_U=0.2, input_shape=(seq_length, 128)))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(128, activation='sigmoid'))
    model.compile(loss='mse', optimizer='adam')

    model.fit(x, y, epochs=2, batch_size=1024)

    # model.save('word2vec.model')

    # model = word2vec.Word2Vec.load('word2vec.model')

    def predict_next(input_array):
        x = np.reshape(input_array, (-1, seq_length, 128))
        y = model.predict(x)
        return y

    def string_to_index(raw_input):
        input_stream = jieba.lcut(raw_input)
        res = []
        for word in input_stream[(len(input_stream) - seq_length):]:
            res.append(word2vec_model[word])
        return np.array(res)

    def y_to_word(y):
        word = word2vec_model.most_similar(positive=y, topn=1)  # 获取单个词相关的前n个词语
        return word

    def generate_article(init, rounds=30):
        in_string = init.lower()
        for i in range(rounds):
            n = y_to_word(predict_next(string_to_index(in_string)))
            in_string += ' ' + n[0][0]
            # print('n[0]:', n[0])  =  ('curiosity', 0.7301754951477051)
            print('n[0][0]:', n[0][0])
        return in_string

    article1 = generate_article('西安碑林区为孩子健康成长创造良好社会环境')
    print(article1)


if __name__ == '__main__':
    # wordvec = WordVec()
    # wordvec.train()
    test()
