import os
import pickle
import pandas as pd
import numpy as np
import jieba
from corpus import chinese_ner_path
from tools import running_of_time

relation2id_path = os.path.join(chinese_ner_path, 'people-relation', 'relation2id.txt')
train_path = os.path.join(chinese_ner_path, 'people-relation', 'train.txt')
train_cut_path = os.path.join(chinese_ner_path, 'people-relation', 'train_cut.txt')

per_samples = 2000
word_max_len = 60


# np.random.seed(1)

class FormatData:
    def __init__(self):
        self.relation2id = dict()
        self.word2id = dict()
        self.id2word = dict()
        self.load_other()

    def get_relation2id(self):
        rel = {}
        with open(relation2id_path, mode='r', encoding='utf8') as f:
            for line in f.readlines():
                k, v = line.strip().split()
                rel[k] = int(v)
        return rel

    def padding_word(self, words):
        ids = []
        for word in words:
            if word in self.word2id:
                ids.append(self.word2id[word])
            else:
                ids.append(self.word2id['unknow'])
        if len(ids) > word_max_len:
            ids = ids[:word_max_len]
        else:
            ids.extend([self.word2id['blank']] * (word_max_len - len(ids)))
        return ids

    def padding_pos(self, pos):
        def _pos(num):
            if num < -40:
                return 0
            if -40 <= num <= 40:
                return num + 40
            if 40 < num:
                return 80

        pos = [_pos(i) for i in pos]
        if len(pos) > word_max_len:
            pos = pos[:word_max_len]
        else:
            pos.extend([81] * (word_max_len - len(pos)))
        return pos

    def cut_text(self, n1, n2, text):
        jieba.add_word(n1, 300)
        jieba.add_word(n2, 300)
        return jieba.cut(text)

    @running_of_time
    def trans_train_data(self):
        m = 0
        with open(train_cut_path, mode='w', encoding='utf8') as w:
            with open(train_path, mode='r', encoding='utf8') as f:
                for line in f.readlines():
                    n1, n2, r, t = line.strip().split()[:4]
                    t = ' '.join(self.cut_text(n1, n2, t))
                    if len(t.split()) > m:
                        m = len(t.split())
                    w.write(' '.join([n1, n2, r, t]) + '\n')
        print('max word len', m)

    @running_of_time
    def save_other(self):
        # 先clear原有字典
        self.word2id.clear()
        self.id2word.clear()
        self.relation2id.clear()

        self.relation2id = self.get_relation2id()
        with open(train_cut_path, mode='r', encoding='utf8') as f:
            for line in f.readlines():
                n1, n2, r, *t = line.strip().split()
                for word in t:
                    if word not in self.word2id:
                        self.word2id[word] = len(self.word2id) + 1
        self.word2id['unknow'] = len(self.word2id) + 1
        self.word2id['blank'] = len(self.word2id) + 1
        self.id2word = dict((v, k) for k, v in self.word2id.items())
        with open(os.path.join(chinese_ner_path, 'word_id.pkl'), mode='wb') as f:
            pickle.dump(self.word2id, f)
            pickle.dump(self.id2word, f)
            pickle.dump(self.relation2id, f)

    @running_of_time
    def load_other(self):
        with open(os.path.join(chinese_ner_path, 'word_id.pkl'), mode='rb') as f:
            self.word2id = pickle.load(f)
            self.id2word = pickle.load(f)
            self.relation2id = pickle.load(f)

    @running_of_time
    def format_data(self):
        def _index(lis, w):
            for idx, word in enumerate(lis):
                for i in range(len(word) - len(w) + 1):
                    if word[i:i + len(w)] == w:
                        return idx
                for c in w:
                    if c in word:
                        return idx

        data = dict()
        output = dict()
        with open(train_cut_path, mode='r', encoding='utf8') as f:
            for line in f.readlines():
                n1, n2, r, *t = line.strip().split()
                if r not in data:
                    data[r], output[r] = [], []
                data[r].append([n1, n2, r, t])

        for key, values in data.items():
            for value in values:
                n1, n2, r, t = value
                sentence, pos1, pos2 = [], [], []
                try:
                    idx_n1, idx_n2 = t.index(n1), t.index(n2)
                except Exception as e:
                    idx_n1, idx_n2 = _index(t, n1), _index(t, n2)
                for idx, word in enumerate(t):
                    sentence.append(word)
                    pos1.append(idx - idx_n1)
                    pos2.append(idx - idx_n2)
                sentence = self.padding_word(sentence)
                pos1 = self.padding_pos(pos1)
                pos2 = self.padding_pos(pos2)
                output[r].append([pos1, pos2, self.relation2id[r], sentence])

        output2 = dict()
        for key, values in output.items():
            if key not in output2:
                output2[key] = []
            location = np.random.permutation(len(values))
            for i in location[:per_samples]:
                output2[key].append(values[i])

        datas = []
        labels = []
        position_e1 = []
        position_e2 = []
        for key, values in output2.items():
            for value in values:
                datas.append(value[3])
                labels.append(value[2])
                position_e1.append(value[1])
                position_e2.append(value[0])

        datas = np.array(datas)
        labels = np.array(labels)
        position_e1 = np.array(position_e1)
        position_e2 = np.array(position_e2)

        self.save(datas, labels, position_e1, position_e2)

    def save(self, datas, labels, position_e1, position_e2):
        with open(os.path.join(chinese_ner_path, 'train_data.pkl'), mode='wb') as f:
            pickle.dump(datas, f)
            pickle.dump(labels, f)
            pickle.dump(position_e1, f)
            pickle.dump(position_e2, f)

    def load(self):
        with open(os.path.join(chinese_ner_path, 'train_data.pkl'), mode='rb') as f:
            datas = pickle.load(f)
            labels = pickle.load(f)
            position_e1 = pickle.load(f)
            position_e2 = pickle.load(f)
        return datas, labels, position_e1, position_e2

    def train_test_split(self, train_split=0.8, shuffle=False):
        if shuffle is True:
            self.format_data()
        datas, labels, position_e1, position_e2 = self.load()
        print('datas shape', datas.shape)
        print('position e1 shape', position_e1.shape)
        print('position e2 shape', position_e2.shape)
        location = np.random.permutation(datas.shape[0])
        idx = int(len(location) * train_split)
        return datas[location][:idx], datas[location][idx:], labels[location][:idx], labels[location][idx:], \
               position_e1[location][:idx], position_e1[location][idx:], \
               position_e2[location][:idx], position_e2[location][idx:]


if __name__ == '__main__':
    f = FormatData()
    f.trans_train_data()
    f.save_other()
    # f.load_other()
    # f.format_data()
    # f.train_test_split(shuffle=True)
