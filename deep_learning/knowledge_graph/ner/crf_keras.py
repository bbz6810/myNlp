""" 源连接https://github.com/stephen-v/zh-NER-keras
    ner+关系抽取 https://blog.csdn.net/NeilGY/article/details/87966676

"""
import os
import platform
import numpy as np
from collections import Counter
from keras import layers, models
from keras.preprocessing.sequence import pad_sequences
from keras_contrib.layers import CRF

from deep_learning.data_pretreatment import NNParam
from corpus import crf_model_path, ner_relation_extract_path
from corpus.load_corpus import LoadCorpus
from keras.initializers import he_normal

chunk_tags = ['O', 'B-PER', 'I-PER', 'B-LOC', 'I-LOC', "B-ORG", "I-ORG"]


def _parse_data(file_handler):
    if platform.system() == 'Windows':
        split_text = '\r\n'
    else:
        split_text = '\n'
    text = file_handler.read().decode('utf8')
    file_handler.close()

    data = [[line.strip() for line in lines.split(split_text)] for lines in
            text.strip().split(split_text + split_text)]
    return data


def _process_data(data, vocab, chunk_tags, maxlen=None, onehot=False):
    if maxlen is None:
        maxlen = max(len(s) for s in data)
    word2idx = dict((k, v) for v, k in enumerate(vocab))
    x = [[word2idx.get(w.split()[0], 1) for w in s] for s in data]
    y = [[chunk_tags.index(w.split()[1]) for w in s] for s in data]
    x = pad_sequences(x, maxlen=maxlen)
    y = pad_sequences(y, maxlen=maxlen, value=-1)
    if onehot:
        y = np.eye(len(chunk_tags), dtype='float32')[y]
    else:
        y = np.expand_dims(y, 2)
    print(x.shape)
    print(y.shape)
    return x, y


def process_data(data, vocab, maxlen=100):
    word2idx = dict((w, i) for i, w in enumerate(vocab))
    x = [word2idx.get(w, 1) for w in data]
    length = len(x)
    x = pad_sequences([x], maxlen=maxlen)
    return x, length


def load_data():
    crf_param = NNParam()
    train = _parse_data(open(os.path.join(ner_relation_extract_path, 'crf-train.txt'), 'rb'))
    test = _parse_data(open(os.path.join(ner_relation_extract_path, 'crf-test.txt'), 'rb'))

    wordcount = Counter(k.split()[0] for sample in train for k in sample)
    d_vocab = dict()
    for k, v in wordcount.items():
        if v >= 2:
            d_vocab[k] = len(d_vocab)
    vocab = [w for w, v in wordcount.items() if v >= 2]

    crf_param.set_embedding_dim(60)
    crf_param.set_vocab_size(len(vocab))

    train = _process_data(train, vocab, chunk_tags)
    test = _process_data(test, vocab, chunk_tags)
    print('加载数据语料完毕.')
    return train, test, vocab, crf_param, d_vocab


class CRFKeras:
    def __init__(self):
        self.nn_param = None

    def build_model(self, embedding_matrix):
        model = models.Sequential()
        model.add(
            layers.Embedding(self.nn_param.vocab_size + 1, self.nn_param.embedding_dim,
                             mask_zero=True))
        model.add(layers.Bidirectional(
            layers.LSTM(self.nn_param.embedding_dim // 2, return_sequences=True,
                        weights=[embedding_matrix], kernel_initializer=he_normal(1),
                        bias_initializer=he_normal(2), kernel_regularizer='l2')))
        crf = CRF(len(chunk_tags), sparse_target=True)
        model.add(crf)
        print(model.summary())
        model.compile(optimizer='adam', loss=crf.loss_function, metrics=[crf.accuracy])
        # model.compile(optimizer='rmsprop', loss=ner.loss_function, metrics=[ner.accuracy])
        return model

    def create_matrix(self, vocab):
        # wv_model = LoadCorpus.load_wv_model()
        wv_model = LoadCorpus.load_wv60_model()
        embedding_matrix = np.zeros(shape=(self.nn_param.vocab_size + 1, self.nn_param.embedding_dim))
        print(embedding_matrix.shape)
        for word, index in vocab.items():
            if word in wv_model:
                embedding_matrix[index] = wv_model[word]
        print("词嵌入的大小", embedding_matrix.shape)
        return embedding_matrix

    def train(self, train=True):
        (train_x, train_y), (test_x, test_y), vocab, self.nn_param, d_vocab = load_data()
        embedding_matrix = self.create_matrix(vocab=d_vocab)
        model = self.build_model(embedding_matrix)
        if train:
            model.fit(train_x, train_y, batch_size=32, epochs=10, validation_data=[test_x, test_y])
            model.save(crf_model_path)
        else:
            model.load_weights(crf_model_path)
        return model, vocab

    def predict(self):
        model, vocab = self.train(train=False)
        predict_text = '中华人民共和国国务院总理周恩来在外交部长陈毅的陪同下，连续访问了埃塞俄比亚等非洲10国以及阿尔巴尼亚'
        s, lens = process_data(predict_text, vocab)

        model.load_weights(crf_model_path)
        raw = model.predict(s)[0][-lens:]
        result = [int(np.argmax(row)) for row in raw]
        result_tag = [chunk_tags[i] for i in result]

        for i, j in zip(predict_text, result_tag):
            print(i, j)

        pre, loc, org = '', '', ''

        for x, t in zip(predict_text, result_tag):
            if t in ('B-PER', 'I-PER'):
                pre += ' ' + x if (t == 'B-PER') else x
            if t in ('B-ORG', 'I-ORG'):
                org += ' ' + x if (t == 'B-ORG') else x
            if t in ('B-LOC', 'I-LOC'):
                loc += ' ' + x if (t == 'B-LOC') else x

        print(['person:' + pre, 'location:' + loc, 'organzation:' + org])


if __name__ == '__main__':
    lstm_crf = CRFKeras()
    lstm_crf.train()
    # ner.predict()
    # load_data()
