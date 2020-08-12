import math
import marshal
import gzip


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
