import re
import gzip
import jieba
import random
import numpy as np
from copy import deepcopy
import subprocess
import platform
import struct
from gensim.models import KeyedVectors, word2vec
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader

from corpus import wv_model_path, news_jieba_path, chatbot100_path, wv60_model_path, xiaohuangji_path, paper_path, \
    mnist_x_train_path, mnist_y_train_path, mnist_x_test_path, mnist_y_test_path
from tools import n_gram, running_of_time, delete_punctuation


class LoadCorpus:
    def __init__(self):
        pass

    @classmethod
    def load_paper_to_word2vec(cls):
        data_list = []
        with open(paper_path, encoding='gb2312', mode='r', errors='ignore') as f:
            for line in f.readlines():
                t = list(map(lambda x: x.split('/')[0], line.split()[1:]))
                # 去除标点符号
                t = [delete_punctuation(i) for i in t if delete_punctuation(i)]
                if t:
                    data_list.append(t)
        return data_list

    @classmethod
    def load_chatbot100_train(cls):
        p = '/Users/zhoubb/projects/corpus/chatbot.txt'
        data = open(chatbot100_path, 'rb')
        output = open(p, 'w', encoding="utf-8")
        lines = data.readlines()
        for line in lines:
            segments = ' '.join(jieba.cut(line.strip()))
            output.write(segments + '\n')
        data.close()
        output.close()

        def generate_XY(segments_file):
            f = open(segments_file, 'r', encoding='utf-8')
            data = f.read()
            X = []
            Y = []
            conversations = data.split('E')
            for q_a in conversations:
                if re.findall('.*M.*M.*', q_a, flags=re.DOTALL):
                    q_a = q_a.strip()
                    q_a_pair = q_a.split('M')
                    X.append(q_a_pair[1].strip())
                    Y.append(q_a_pair[2].strip())
            f.close()
            return X, Y

        return generate_XY(p)

    @classmethod
    def load_xiaohuangji_train(cls):
        p = '/Users/zhoubb/projects/corpus/xiaohuangji50w_fenciA.txt'
        data = open(xiaohuangji_path, 'rb')
        output = open(p, 'w', encoding="utf-8")
        lines = data.readlines()
        for line in lines:
            output.write(line.replace(b'/', b' ').decode())
        output.close()
        data.close()

        def generate_XY(segments_file):
            f = open(segments_file, 'r', encoding='utf-8')
            data = f.read()
            X = []
            Y = []
            conversations = data.split('E')
            for q_a in conversations:
                if re.findall('.*M.*M.*', q_a, flags=re.DOTALL):
                    q_a = q_a.strip()
                    q_a_pair = q_a.split('M')
                    X.append(q_a_pair[1].strip())
                    Y.append(q_a_pair[2].strip())
            f.close()
            print('x对话最长字符', max(len(i) for i in X))
            print('x对话平均字符', sum(len(i) for i in X) / len(X))
            print('y对话最长字符', max(len(i) for i in Y))
            print('y对话平均字符', sum(len(i) for i in Y) / len(Y))
            return X, Y

        return generate_XY(p)

    @classmethod
    def load_news_train(cls, _n_gram=None, c=2):
        """new_type = ['ent', 'edu', 'economic', 'house', 'game', 'stock', 'affairs', 'constellation', 'fashion', 'home',
                    'sports', 'science', 'lottery']
            # new_dict = dict((k, 0) for k in new_type)
        :param _n_gram:
        :param c: 默认加载所有语料
        :return:
        """

        load_labels = [
            'ent', 'edu', 'economic', 'house', 'game', 'stock', 'affairs', 'constellation', 'fashion', 'home', 'sports',
            'science', 'lottery']

        random.shuffle(load_labels)
        load_label = load_labels[:c]
        # load_label = ['fashion', 'ent']
        print('随机取{}个标签为:{}'.format(c if c < len(load_labels) else len(load_labels), load_label))

        train_x = []
        train_y = []
        with open(news_jieba_path, encoding='gb2312', errors='ignore', mode='r') as f:
            if _n_gram:
                for line in f.readlines():
                    t = line.strip().split(' ')
                    x, y = t[:-1], t[-1]
                    x = n_gram(x, _n_gram)
                    if y not in load_label:
                        continue
                    train_x.append(' '.join(x))
                    train_y.append(y)
            else:
                for line in f.readlines():
                    t = line.strip().split(' ')
                    x, y = t[:-1], t[-1]
                    if y not in load_label:
                        continue
                    train_x.append(' '.join(x))
                    train_y.append(y)
        return train_x, train_y

    @classmethod
    @running_of_time
    def load_wv_model(cls):
        """加载人民日报语料生成的word2vec模型

        :return:
        """
        return KeyedVectors.load_word2vec_format(wv_model_path)

    @classmethod
    @running_of_time
    def load_wv60_model(cls):
        print('加载模型', wv60_model_path)
        return word2vec.Word2Vec.load(wv60_model_path)

    @classmethod
    @running_of_time
    def load_mnist(cls, data_name=mnist_x_train_path, label_name=mnist_y_train_path, test_name=mnist_x_test_path,
                   test_label=mnist_y_test_path):
        mnist = Mnist(data_name, label_name, test_name, test_label)
        x_train = deepcopy(mnist.train_set)
        y_train = deepcopy(mnist.train_labels)
        x_test = deepcopy(mnist.test_set)
        y_test = deepcopy(mnist.test_label)
        # return x_train, y_train, x_test, y_test

        # x_train = np.concatenate([x_train[y_train == 0], x_train[y_train == 1]])
        # y_train = np.concatenate(
        #     [np.zeros(shape=(list(y_train == 0).count(True),)), np.ones(shape=(list(y_train == 1).count(True),))])
        #
        # x_test = np.concatenate([x_test[y_test == 0], x_test[y_test == 1]])
        # y_test = np.concatenate(
        #     [np.zeros(shape=(list(y_test == 0).count(True),)), np.ones(shape=(list(y_test == 1).count(True),))]
        # )
        # # return x_train / 255, y_train, x_test / 255, y_test

        return (x_train > 128) + 0, y_train, (x_test > 128) + 0, y_test


class Mnist(Dataset):
    def __init__(self, data_name, label_name, test_name, test_label, transform=None):
        (train_set, train_labels, test_set, test_label) = self.load_data(data_name, label_name, test_name, test_label)
        self.train_set = train_set
        self.train_labels = train_labels
        self.test_set = test_set
        self.test_label = test_label
        self.transform = transform

    def __getitem__(self, item):
        img, target = self.train_set[item], int(self.train_labels[item])
        if self.transform is not None:
            img = self.transform(img)
        return img, target

    def __len__(self):
        return len(self.train_set)

    def load_data(self, data_name, lable_name, test_name, test_label):
        with gzip.open(lable_name, mode='rb') as lbpath:
            y_train = np.frombuffer(lbpath.read(), np.uint8, offset=8)

        with gzip.open(data_name, mode='rb') as imgpath:
            x_train = np.frombuffer(imgpath.read(), np.uint8, offset=16).reshape(len(y_train), 28 * 28)

        with gzip.open(test_label, mode='rb') as test_y:
            y_test = np.frombuffer(test_y.read(), np.uint8, offset=8)

        with gzip.open(test_name, mode='rb') as test_x:
            x_test = np.frombuffer(test_x.read(), np.uint8, offset=16).reshape(len(y_test), 28 * 28)

        return x_train, y_train, x_test, y_test


if __name__ == '__main__':
    mnist = LoadCorpus.load_mnist(mnist_x_train_path, mnist_y_train_path, mnist_x_test_path, mnist_y_test_path)
