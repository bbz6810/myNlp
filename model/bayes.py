import math
import marshal
import gzip
import numpy as np
from corpus.load_corpus import LoadCorpus


class AddOneCategory:
    def __init__(self):
        self.total = 0
        self.d = {}
        self.none = 1

    def add(self, key, value):
        self.total += value
        if key not in self.d:
            self.d[key] = 1
            self.total += 1
        self.d[key] += value


class Bayes:
    def __init__(self):
        self.total = 0
        self.category_dict = {}

    def save(self, fname, iszip=True):
        fname = fname + '.bayes'
        d = {}
        d['total'] = self.total
        d['d'] = {}
        for k, v in self.category_dict.items():
            d['d'][k] = v.__dict__
        if not iszip:
            marshal.dump(d, open(fname, 'wb'))
        else:
            with gzip.open(fname, 'wb') as f:
                f.write(marshal.dumps(d))

    def load(self, fname, iszip=True):
        fname = fname + '.bayes'
        if not iszip:
            d = marshal.load(open(fname, 'rb'))
        else:
            with gzip.open(fname, 'rb') as f:
                d = marshal.loads(f.read())
        self.total = d['total']
        self.category_dict = {}
        for k, v in d['d'].items():
            self.category_dict[k] = AddOneCategory()
            self.category_dict[k].__dict__ = v

    def train(self, data):
        for d in data:
            c = d[1]
            if c not in self.category_dict:
                self.category_dict[c] = AddOneCategory()
            for word in d[0]:
                self.category_dict[c].add(word, 1)
        self.total = sum(map(lambda x: self.category_dict[x].total, self.category_dict.keys()))

    def predict(self, words):
        tmp = {}
        for k in self.category_dict:
            tmp[k] = math.log(self.category_dict[k].total) - math.log(self.total)
            for word in words:
                tmp[k] += math.log(self.category_dict[k].d.get(word, 1))
        ret, prob = 0, 0
        for k in self.category_dict:
            now = 0
            for kk in self.category_dict:
                now += math.exp(tmp[kk] - tmp[k])
            try:
                now = 1 / now
            except OverflowError:
                now = 0
            if now > prob:
                ret, prob = k, now
        return ret, prob


def naive_bayes(py, pxy, x):
    feature_num = 768
    class_num = 10
    p = [0] * class_num
    for i in range(class_num):
        sum = 0
        for j in range(feature_num):
            sum += pxy[i][j][x[j]]
        p[i] = sum + py[i]
    return p.index(max(p))


def model_test(py, pxy, test_x, test_y):
    error_cnt = 0
    for i in range(test_x.shape[0]):
        predict = naive_bayes(py, pxy, test_x[i])
        if predict != test_y[i]:
            error_cnt += 1
    return 1 - error_cnt / test_x.shape[0]


def get_all_probability(train_x, train_y):
    feature_num = 768
    class_num = 10

    py = np.zeros((class_num, 1))
    for i in range(class_num):
        py[i] = ((np.sum(np.mat(train_x) == i)) + 1) / (train_x.shape[0] + 10)
    py = np.log(py)

    pxy = np.zeros((class_num, feature_num, 2))
    for i in range(train_x.shape[0]):
        label = train_y[i]
        x = train_x[i]
        for j in range(feature_num):
            pxy[label][j][x[j]] += 1

    for label in range(class_num):
        for j in range(feature_num):
            pxy0 = pxy[label][j][0]
            pxy1 = pxy[label][j][1]
            pxy[label][j][0] = np.log((pxy0 + 1) / (pxy0 + pxy1 + 2))
            pxy[label][j][1] = np.log((pxy1 + 1) / (pxy0 + pxy1 + 2))
    return py, pxy


if __name__ == '__main__':
    train_x, train_y, test_x, test_y = LoadCorpus.load_mnist()
    py, pxy = get_all_probability(train_x, train_y)
    score = model_test(py, pxy, test_x, test_y)
    print('score', score)
