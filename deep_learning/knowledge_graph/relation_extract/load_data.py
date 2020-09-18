import os
import pickle
import pandas as pd
import numpy as np
from collections import deque
from corpus import chinese_ner_path
from tools import running_of_time

relation2id_path = os.path.join(chinese_ner_path, 'people-relation', 'relation2id.txt')
train_path = os.path.join(chinese_ner_path, 'people-relation', 'train.txt')
word_max_len = 50


class FormatData:
    def __init__(self):
        self.relation2id = self.get_relation2id()
        self.datas = deque()
        self.labels = deque()
        self.position_e1 = deque()
        self.position_e2 = deque()
        self.count = [0] * len(self.relation2id)
        self.total_data = 0
        self.word2id = dict()
        self.id2word = dict()

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
            if num <= -40:
                return 0
            elif -40 < num < 40:
                return num + 40
            elif 40 <= num:
                return 80

        pos = [_pos(i) for i in pos]
        if len(pos) > word_max_len:
            pos = pos[:word_max_len]
        else:
            pos.extend([81] * (len(pos) - word_max_len))
        return pos

    @running_of_time
    def format_train_data(self):
        with open(train_path, mode='r', encoding='utf8') as f:
            for line in f.readlines():
                n1, n2, r, t = line.strip().split()[:4]

                for word in t:
                    if word not in self.word2id:
                        self.word2id[word] = len(self.word2id) + 1
                if self.count[self.relation2id[r]] < 1500:
                    sentence = []
                    pos1, pos2 = [], []
                    nindex1 = t.index(n1)
                    nindex2 = t.index(n2)
                    for idx, word in enumerate(t):
                        sentence.append(word)
                        pos1.append(idx - nindex1)
                        pos2.append(idx - nindex2)
                    self.datas.append(sentence)
                    self.labels.append(self.relation2id[r])
                    self.position_e1.append(pos1)
                    self.position_e2.append(pos2)

                self.count[self.relation2id[r]] += 1
                self.total_data += 1

        self.word2id['unknow'] = len(self.word2id) + 1
        self.word2id['blank'] = len(self.word2id) + 1
        self.id2word = dict((v, k) for k, v in self.word2id.items())
        print('各类文本行数', self.count)
        print('总共文本行数', self.total_data)
        print('用到文本行数', len(self.datas))

        df_data = pd.DataFrame({'datas': self.datas, 'labels': self.labels, 'position_e1': self.position_e1,
                                'position_e2': self.position_e2})

        df_data['datas'] = df_data['datas'].apply(self.padding_word)
        df_data['position_e1'] = df_data['position_e1'].apply(self.padding_pos)
        df_data['position_e2'] = df_data['position_e2'].apply(self.padding_pos)

        self.datas = df_data['datas'].values
        self.position_e1 = df_data['position_e1'].values
        self.position_e2 = df_data['position_e2'].values

    def save_train(self):
        with open(os.path.join(chinese_ner_path, 'train_data.pkl'), mode='wb') as f:
            pickle.dump(self.datas, f)
            pickle.dump(self.labels, f)
            pickle.dump(self.position_e1, f)
            pickle.dump(self.position_e2, f)
            pickle.dump(self.relation2id, f)
            pickle.dump(self.word2id, f)
            pickle.dump(self.id2word, f)

    def load_train(self):
        with open(os.path.join(chinese_ner_path, 'train_data.pkl'), mode='rb') as f:
            self.datas = pickle.load(f)
            self.labels = pickle.load(f)
            self.position_e1 = pickle.load(f)
            self.position_e2 = pickle.load(f)
            self.relation2id = pickle.load(f)
            self.word2id = pickle.load(f)
            self.id2word = pickle.load(f)

    @running_of_time
    def format_test_data(self):
        with open(train_path, mode='r', encoding='utf8') as f:
            for line in f.readlines():
                n1, n2, r, t = line.strip().split()[:4]

                for word in t:
                    if word not in self.word2id:
                        self.word2id[word] = len(self.word2id) + 1
                if 1500 < self.count[self.relation2id[r]] < 1800:
                    sentence = []
                    pos1, pos2 = [], []
                    nindex1 = t.index(n1)
                    nindex2 = t.index(n2)
                    for idx, word in enumerate(t):
                        sentence.append(word)
                        pos1.append(idx - nindex1)
                        pos2.append(idx - nindex2)
                    self.datas.append(sentence)
                    self.labels.append(self.relation2id[r])
                    self.position_e1.append(pos1)
                    self.position_e2.append(pos2)

                self.count[self.relation2id[r]] += 1

        print('各类文本行数', self.count)
        print('总共文本行数', self.total_data)
        print('用到文本行数', len(self.datas))

        df_data = pd.DataFrame({'datas': self.datas, 'labels': self.labels, 'position_e1': self.position_e1,
                                'position_e2': self.position_e2})

        df_data['datas'] = df_data['datas'].apply(self.padding_word)
        df_data['position_e1'] = df_data['position_e1'].apply(self.padding_pos)
        df_data['position_e2'] = df_data['position_e2'].apply(self.padding_pos)

        self.datas = df_data['datas'].values
        self.position_e1 = df_data['position_e1'].values
        self.position_e2 = df_data['position_e2'].values

    def save_test(self):
        with open(os.path.join(chinese_ner_path, 'train_data.pkl'), mode='wb') as f:
            pickle.dump(self.datas, f)
            pickle.dump(self.labels, f)
            pickle.dump(self.position_e1, f)
            pickle.dump(self.position_e2, f)

    def load_test(self):
        with open(os.path.join(chinese_ner_path, 'train_data.pkl'), mode='rb') as f:
            self.datas = pickle.load(f)
            self.labels = pickle.load(f)
            self.position_e1 = pickle.load(f)
            self.position_e2 = pickle.load(f)


if __name__ == '__main__':
    f = FormatData()
    f.format_train_data()
    f.save_train()
