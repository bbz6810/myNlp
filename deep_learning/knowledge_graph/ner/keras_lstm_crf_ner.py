import os
import time
import platform
import numpy as np
from keras import layers, models
from keras_contrib.layers import CRF
from keras.callbacks import ModelCheckpoint, EarlyStopping

from corpus import crf_model_path

from collections import Counter
from keras.preprocessing.sequence import pad_sequences
import pickle

# data_path = r'D:\projects\zh-NER-keras-master\data'
data_path = r'D:\code_data\ner\命名实体识别样本\data-0928-train300\bio'
train_file = 'train_bio_200.txt'
test_file = 'test_bio.txt'
# train_file = 'train_bio_1.txt'
# test_file = 'test_bio_1.txt'
chunk_tags = []


def load_data():
    train_x, train_y = _parse_data(open(os.path.join(data_path, train_file), mode='r', encoding='utf-8'))
    test_x, test_y = _parse_data(open(os.path.join(data_path, test_file), mode='r', encoding='utf-8'))

    # todo
    s = set()
    for i in train_y:
        s.update(set(i))
    for j in test_y:
        s.update(set(j))
    global chunk_tags
    chunk_tags = list(s)

    word_counts = Counter(row[0].lower() for sample in train_x for row in sample)
    vocab = [w for w, f in iter(word_counts.items()) if f >= 2]
    print('vocab size', len(vocab))
    # save initial config data
    with open('model/config.pkl', 'wb') as outp:
        pickle.dump((vocab, chunk_tags), outp)

    train = _process_data(train_x, train_y, vocab, chunk_tags)
    test = _process_data(test_x, test_y, vocab, chunk_tags)
    return train, test, (vocab, chunk_tags)


def _parse_data(fh):
    data_x = []
    data_y = []
    row_x = []
    row_y = []
    idx = 0
    for line in fh:
        idx += 1
        if idx % 1000000 == 0:
            print(idx, fh)
        line = line.strip().split(' ')
        if len(line) == 2:
            row_x.append(line[0])
            row_y.append(line[1])
        else:
            data_x.append(row_x)
            data_y.append(row_y)
            row_x = []
            row_y = []
    return data_x, data_y


def _process_data(data_x, data_y, vocab, chunk_tags, maxlen=None, onehot=False):
    if maxlen is None:
        maxlen = max(len(s) for s in data_x)
        print('max len', maxlen)
    word2idx = dict((w, i) for i, w in enumerate(vocab))
    x = [[word2idx.get(w.lower(), 1) for w in s] for s in data_x]  # set to <unk> (index 1) if not in vocab

    y_chunk = [[chunk_tags.index(w) for w in s] for s in data_y]

    x = pad_sequences(x, maxlen)  # left padding

    y_chunk = pad_sequences(y_chunk, maxlen, value=-1)

    if onehot:
        y_chunk = np.eye(len(chunk_tags), dtype='float32')[y_chunk]
    else:
        y_chunk = np.expand_dims(y_chunk, 2)
    return x, y_chunk


def process_data(data, vocab, maxlen=100):
    word2idx = dict((w, i) for i, w in enumerate(vocab))
    x = [word2idx.get(w, 1) for w in data]
    length = len(x)
    x = pad_sequences([x], maxlen=maxlen)
    return x, length


class CRFKeras:
    def __init__(self):
        self.nn_param = None

    def build_model(self, vocab_size):
        model = models.Sequential()
        model.add(
            layers.Embedding(vocab_size + 1, 128, mask_zero=True))
        model.add(layers.Bidirectional(layers.LSTM(256 // 2, return_sequences=True)))
        crf = CRF(len(chunk_tags), sparse_target=True)
        model.add(crf)
        print(model.summary())
        model.compile(optimizer='adam', loss=crf.loss_function, metrics=[crf.accuracy])
        return model

    def calls(self):
        callback = ModelCheckpoint(
            filepath='./model/ner_{}.model'.format(int(time.time())), monitor='roc_auc',
            verbose=1, save_best_only=True, mode='max', period=2)

    def callbacks(self):
        callback = ModelCheckpoint(filepath='./model/ner_{}.model'.format(int(time.time())), monitor='roc_auc',
                                   verbose=1, save_best_only=True, mode='max', period=2)
        earlystop = EarlyStopping(monitor='val_acc', patience=2, verbose=1, mode='max')
        return [callback, earlystop]

    def train(self, train=True):
        (train_x, train_y), (test_x, test_y), (vocab, d_vocab) = load_data()
        print('train x shape', train_x.shape)
        print('train y shape', train_y.shape)
        model = self.build_model(vocab_size=len(vocab))
        if train:
            model.fit(train_x, train_y, epochs=10, validation_data=[test_x, test_y], callbacks=self.callbacks())
            model.save('./model/ner.m')
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
    # _parse_data(open(os.path.join(data_path, train_file), mode='r', encoding='utf-8'))
