import os
import re
import time
import numpy
import jieba
from hashlib import md5
from copy import deepcopy
from gensim.models import KeyedVectors
from corpus import stop_path

stop = set(' ')

wechat_path = '/Users/zhoubb/Downloads/news_fasttext_train.txt'
wechat_jieba_path = '/Users/zhoubb/Downloads/news_fasttext_train_jieba.txt'
from corpus import paper_wv_path

with open(stop_path, encoding='utf8', mode='r') as f:
    for line in f.readlines():
        stop.add(line.strip())


def filter_stop(words):
    """过滤停用词

    :param words:
    :return:
    """
    return list(filter(lambda x: x not in stop, words))


def fetch_file_path(path):
    """获取该路径下的所有文件

    :param path:
    :return:
    """
    return [file for file in os.listdir(path)]


def sigmoid(wx):
    """sigmoid函数

    :param wx:
    :return:
    """
    return 1.0 / (1.0 + numpy.exp(-wx))


def split_train_test(x, y, p=0.8):
    # numpy.random.shuffle(x)
    # numpy.random.shuffle(y)
    x_shape = x.shape
    y_shape = y.shape
    return x[:int(x_shape[0] * p)], y[:int(y_shape[0] * p)], x[int(x_shape[0] * p):], y[int(y_shape[0] * p):]


def md5_str(s):
    m = md5(s.encode('utf-8'))
    return m.hexdigest()


def hash2bin(s):
    """hash成二进制字符

    :param s:
    :return:
    """
    t = hash(s)
    t = t if t > 0 else -t
    return bin(t).replace('0b', '').zfill(64)


def hamming(n1, n2):
    """计算两个int数的汉明距离

    :param n1:
    :param n2:
    :return:
    """
    return bin(n1 ^ n2).count('1')


def print_dict(d, black_len):
    """打印字典

    :param d:
    :param black_len:
    :return:
    """
    d = dict(sorted(d.items(), key=lambda x: isinstance(x[1], dict), reverse=True))
    for k, v in d.items():
        if isinstance(v, dict):
            print('{}key:{}, value:'.format('\t' * black_len, k))
            print_dict(v, black_len + 1)
        else:
            print('{}key:{}, value:{}'.format('\t' * black_len, k, v))


def delete_punctuation(s):
    """去除标点符号

    :param s:
    :return:
    """
    d = '~`!#$%^&*()_+-=|\';":/.,?><~·！@#￥%……&*（）——+-=“：’；、。，？》《{}．‘／＞［］〉〈・”'
    return re.sub('[{}]+'.format(d), '', s)


def n_gram(sen, n=2):
    """n元语法拆分

    :param sen:
    :param n:
    :return:
    """
    sen = list(map(str, sen))
    retv = deepcopy(sen)
    for k in range(2, n + 1):
        for i in range(len(sen) - k + 1):
            retv.append(''.join(sen[i:i + k]))
    return retv


def wash_news():
    """清洗新闻分类语料

    :return:
    """
    now = time.time()
    ff = open(wechat_jieba_path, encoding='gb2312', errors='ignore', mode='w')
    with open(wechat_path, encoding='utf8', errors='ignore') as f:
        for index, i in enumerate(f.readlines()):
            k, lab = i.strip().split('__label__')
            ts = ' '.join(
                filter_stop(jieba.cut(''.join(map(lambda x: x.strip().replace(' ', ''), k))))) + ' {}\n'.format(lab)
            if index % 1000 == 0:
                print(index, time.time() - now)
            ff.write(ts)
            ff.flush()
    ff.close()


def running_of_time(f):
    def g(*args, **kwargs):
        now = time.time()
        r = f(*args, **kwargs)
        print('function: {}, cost time: {}'.format(f.__name__, time.time() - now))
        return r

    return g
