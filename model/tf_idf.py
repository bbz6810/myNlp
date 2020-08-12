import math
import marshal
import gzip
import pickle

import numpy as np


class TFIDF:
    def __init__(self):
        self.labels = []
        self.word_index = {}

        self.np_tf_list = None
        self.np_df = None
        self.tf_list = None
        self.idf = None
        self.tf_idf = None

    def save(self, fname, iszip=True):
        fname = fname + '.tfidf'
        d = {
            'np_tf_list': pickle.dumps(self.np_tf_list),
            'tf_list': pickle.dumps(self.tf_list),
            'np_df': pickle.dumps(self.np_df),
            'idf': pickle.dumps(self.idf),
            'labels': pickle.dumps(self.labels),
            'word_index': pickle.dumps(self.word_index),
            'tf_idf': pickle.dumps(self.tf_idf)
        }

        # np.save('np_tf_list', self.np_tf_list)

        if not iszip:
            marshal.dump(d, open(fname, 'wb'))
        else:
            with gzip.open(fname, 'wb') as f:
                f.write(marshal.dumps(d))

    def load(self, fname, iszip=True):
        fname = fname + '.tfidf'
        if not iszip:
            d = marshal.load(open(fname, 'rb'))
        else:
            with gzip.open(fname, 'rb') as f:
                d = marshal.loads(f.read())
        for key, value in d.items():
            setattr(self, key, pickle.loads(value))
        # self.__dict__ = d

    def gen_word_index(self, data):
        idx = 0
        # 确定行向量长度
        words = []
        for index, row in enumerate(data):
            tmp = {}
            word, c = row
            self.labels.append(c)
            for w in word:
                tmp[w] = tmp.get(w, 0) + 1
                if w not in self.word_index:
                    self.word_index[w] = idx
                    idx += 1
            words.append(tmp)
        return words

    def init_np_tf_df(self, words_len, vec_len):
        self.np_tf_list = np.array(np.zeros(shape=(words_len, vec_len), dtype='float32'))
        self.np_df = np.array(np.zeros(shape=(vec_len,), dtype='float32'))

    def init_tf(self, words):
        for index, word_dict in enumerate(words):
            for k, v in word_dict.items():
                self.np_tf_list[index][self.word_index[k]] = v

    def init_df(self):
        for i in range(self.np_tf_list.shape[1]):
            self.np_df[i] = len(list(filter(lambda x: x > 0, self.np_tf_list[:, [i]])))

    def calc_tf(self):
        self.tf_list = np.array(self.np_tf_list)
        for index in range(self.tf_list.shape[0]):
            row_total_tf = sum(self.tf_list[index])
            for j in range(self.tf_list.shape[1]):
                self.tf_list[index][j] /= row_total_tf

    def calc_idf(self):
        df_total = sum(self.np_df)
        self.idf = np.array(self.np_df)
        for i in range(self.idf.shape[0]):
            self.idf[i] = math.log((df_total - sum(self.np_tf_list[:, [i]]) + 0.5) / (self.idf[i] + 0.5))

    def calc_tf_idf(self):
        self.tf_idf = np.array(self.tf_list)
        for index in range(self.tf_idf.shape[0]):
            self.tf_idf[[index], :] = self.tf_idf[index] * self.idf

    def train(self, data):
        words = self.gen_word_index(data=data)
        self.init_np_tf_df(len(data), len(self.word_index))

        self.init_tf(words)
        self.init_df()

        self.calc_tf()
        self.calc_idf()
        self.calc_tf_idf()


if __name__ == '__main__':
    tf = TFIDF()
