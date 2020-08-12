from copy import deepcopy
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

from corpus.load_corpus import LoadCorpus


class NNParam:
    def __init__(self):
        self.vocab_size = 10000
        # 利用预训练好的词向量
        self.embedding_dim = 300
        self.max_words = 200
        self.class_num = 2

    def set_vocab_size(self, vocab_size):
        self.vocab_size = vocab_size

    def set_embedding_dim(self, embedding_dim):
        self.embedding_dim = embedding_dim

    def set_max_words(self, max_words):
        self.max_words = max_words

    def set_class_num(self, class_num):
        self.class_num = class_num


class Pretreatment:
    def __init__(self):
        self.tokenizer = Tokenizer()
        self.nnparam = NNParam()
        super(Pretreatment, self).__init__()

    def pre_x(self, x):
        self.tokenizer.fit_on_texts(x)
        self.nnparam.set_vocab_size(len(self.tokenizer.word_index) + 1)
        texts_seq = self.tokenizer.texts_to_sequences(x)
        print('pre_x, 最大长度', max(len(i) for i in texts_seq))
        print('pre_x, 平均长度', sum(len(i) for i in texts_seq) / len(texts_seq))
        self.nnparam.set_max_words(int(sum(len(i) for i in texts_seq) / len(texts_seq)))
        return pad_sequences(texts_seq, maxlen=self.nnparam.max_words)

    def pre_y(self, y):
        set_label = set(map(str, y))
        _y = dict()
        for i in set_label:
            _y[i] = len(_y)

        train_y = list()
        if len(set_label) == 2:
            self.nnparam.set_class_num(2)
            for b in y:
                train_y.append([_y[b]])
        else:
            self.nnparam.set_class_num(len(set_label))
            for b in y:
                t = np.zeros(len(set_label))
                t[_y[b]] = 1
                train_y.append(t)
        train_y = np.array(train_y)
        print('pre_y, 生成y值向量', train_y.shape)
        return train_y

    def train_test_split(self, c, test_size=0.2):
        x, y = self.shuffle(*self.pre_data(c))
        train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=test_size)
        return train_x, test_x, train_y, test_y

    def shuffle(self, x, y):
        location = np.random.permutation(len(x))
        return x[location], y[location]

    def pre_data(self, c):
        x, y = LoadCorpus.load_news_train(c=c)
        train_x, train_y = self.pre_x(x), self.pre_y(y)
        return train_x, train_y

    def create_embedding_matrix(self, split):
        x_r = sorted(self.tokenizer.word_counts.items(), key=lambda x: x[1], reverse=True)
        print("one hot向量长度", len(x_r))
        small_word_index = deepcopy(self.tokenizer.word_index)
        print('该切片的第一个词频次', x_r[split])
        for item in x_r[split:]:
            small_word_index.pop(item[0])
        wv_model = LoadCorpus.load_wv_model()
        embedding_matrix = np.zeros(shape=(self.nnparam.vocab_size + 1, self.nnparam.embedding_dim))
        for word, index in small_word_index.items():
            try:
                embedding_matrix[index] = wv_model[word]
            except Exception as e:
                pass
        print("词嵌入的大小", embedding_matrix.shape)
        return embedding_matrix


if __name__ == '__main__':
    p = Pretreatment()
    x1, x2, y1, y2 = p.train_test_split(c=2, test_size=0.2)
    print(x1.shape)
    print(x2.shape)
    print(y1.shape)
    print(y2.shape)
