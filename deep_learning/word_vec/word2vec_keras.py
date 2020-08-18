# 流程图 https://blog.csdn.net/qq_34862636/article/details/103834058

import marshal
import gzip
import numpy as np
from keras import models, layers
import keras.backend as K
from keras.utils import plot_model

from corpus.load_corpus import LoadCorpus
from corpus import word2vec_model_path, corpus_root_path

word_size = 128  # 词向量大小
window = 5  # 窗口大小
nb_negative = 16  # 随机负采样的样本数
min_count = 10  # 频数小于min_count的将被过滤
nb_worker = 4  # 读取数据的并发数
nb_epoch = 2
subsample_t = 1e-5  # 词频大于subsample_t的词语，会被将采样
nb_sentence_per_batch = 20  # 目前是以句子为单位作为batch


class Word2Vector:
    def __init__(self):
        self.nb_word = None
        self.words = None
        self.id2word = None
        self.word2id = None
        self.train_x = []
        self.train_y = []
        self.wv = None

    def pre_data(self):
        words = dict()
        nb_sentence, total = 0, 0
        sentences = LoadCorpus.load_paper_to_word2vec()

        for sentence in sentences:
            nb_sentence += 1
            for w in sentence:
                if w not in words:
                    words[w] = 0
                words[w] += 1
                total += 1
        print('总词数', total)
        print('句子长度', nb_sentence)

        # 截断小词频单词
        self.words = dict((i, j) for i, j in words.items() if j >= min_count)
        # 词表映射 0 表示UNK
        self.id2word = dict((i + 1, k) for i, k in enumerate(words))
        self.word2id = dict((v, k) for k, v in self.id2word.items())
        self.nb_word = len(words) + 1

        print('词向量大小', self.nb_word)

        subsamples = dict((i, j / total) for i, j in words.items() if j / total > subsample_t)  # 词频
        # 这个降采样公式，是按照word2vec的源码来的
        subsamples = dict((i, subsample_t / j + (subsample_t / j) ** 0.5) for i, j in subsamples.items())
        # 降采样表
        subsamples = dict((self.word2id[i], j) for i, j in subsamples.items() if j < 1.0)

        for sentence in sentences:
            # 构造当前句子的文本向量
            sentence = [0] * window + [self.word2id[w] for w in sentence if w in self.word2id] + [0] * window
            r = np.random.random(len(sentence))
            for i in range(window, len(sentence) - window):
                if sentence[i] in subsamples:
                    self.train_x.append(sentence[i - window:i] + sentence[i + 1:i + 1 + window])
                    self.train_y.append([sentence[i]])
        x, y = np.array(self.train_x), np.array(self.train_y)
        print('样本量x', x.shape)
        print('样本量y', y.shape)
        return x, y

    def build_model(self):
        input_words = layers.Input(shape=(window * 2,), dtype='int32')
        input_vecs = layers.Embedding(self.nb_word, word_size, name='word2vec')(input_words)
        input_vecs_sum = layers.Lambda(lambda x: K.sum(x, axis=1))(input_vecs)

        # 构造负采样
        target_word = layers.Input(shape=(1,), dtype='int32')
        negatives = layers.Lambda(
            lambda x: K.random_uniform((K.shape(x)[0], nb_negative), 0, self.nb_word, dtype='int32'))(
            target_word)
        samples = layers.Lambda(lambda x: K.concatenate(x))([target_word, negatives])

        # 只在抽样内做Dense和softmax
        softmax_weights = layers.Embedding(self.nb_word, word_size, name='w')(samples)
        softmax_biases = layers.Embedding(self.nb_word, 1, name='b')(samples)
        softmax = layers.Lambda(lambda x: K.softmax((K.batch_dot(x[0], K.expand_dims(x[1], 2)) + x[2])[:, :, 0]))(
            [softmax_weights, input_vecs_sum, softmax_biases])
        model = models.Model(inputs=[input_words, target_word], outputs=softmax)
        model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        print(model.summary())
        return model

    def train(self):
        x, y = self.pre_data()
        model = self.build_model()
        model.fit([x, y], np.zeros(shape=(len(x), 1)), batch_size=64, epochs=2, validation_split=0.2)
        self.wv = model.get_weights()[0]
        self.wv = self.wv / (self.wv ** 2).sum(axis=1).reshape((-1, 1)) ** 0.5
        self.save(word2vec_model_path, model.get_weights()[0])

    def most_similarity(self, w, top_k=10):
        self.load(word2vec_model_path)
        if w not in self.word2id:
            print('没有该词')
            return []
        v = self.wv[self.word2id[w]]
        sims = np.dot(self.wv, v)
        sort = sims.argsort()[::-1]
        sort = sort[sort > 0]
        return [(self.id2word[i], sims[i]) for i in sort[:top_k]]

    def sim(self, w1, w2):
        if w1 not in self.word2id:
            print('没有', w1)
            return
        if w2 not in self.word2id:
            print('没有', w2)
            return
        v1 = self.wv[self.word2id[w1]]
        v2 = self.wv[self.word2id[w2]]
        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

    def save(self, fname, wv, iszip=True):
        fname_1 = fname + '.word2vector.dict'
        fname_2 = fname + '.word2vector.wv'
        d = {
            'nb_word': self.nb_word,
            'words': self.words,
            'id2word': self.id2word,
            'word2id': self.word2id
        }
        if not iszip:
            marshal.dump(d, open(fname_1, 'wb'))
        else:
            with gzip.open(fname_1, 'wb') as f:
                f.write(marshal.dumps(d))
        np.save(fname_2, wv)

    def load(self, fname, iszip=True):
        fname_1 = fname + '.word2vector.dict'
        fname_2 = fname + '.word2vector.wv.npy'
        if not iszip:
            d = marshal.load(open(fname_1, 'rb'))
        else:
            with gzip.open(fname_1, 'rb') as f:
                d = marshal.loads(f.read())
        self.nb_word = d['nb_word']
        self.words = d['words']
        self.id2word = d['id2word']
        self.word2id = d['word2id']
        self.wv = np.load(fname_2)

    def view_model(self):
        model = self.build_model()
        plot_model(model, to_file=corpus_root_path + '/word2vec.model.png')


if __name__ == '__main__':
    word2vec = Word2Vector()
    word2vec.train()
    # word2vec.view_model()
    # a = word2vec.most_similarity('男')
    # print(a)
    # b = word2vec.sim('男', '女')
    # print(b)
