import os
import jieba
import random

from gensim.models import word2vec
from gensim import models

from corpus import wv_model_path, news_jieba_path
from tools import n_gram, running_of_time


class LoadCorpus:
    def __init__(self):
        pass

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
        print('随机取{}个标签为:{}'.format(c, load_label))

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
    def load_wv_model(cls, re_build=False):
        """加载目前所有语料生成的word2vec模型
            1、如果没有该文件则生成
            2、如果有该文件则直接加载
            3、如果re_build是True则重新训练词向量

        :return:
        """
        if re_build:
            print('===============begin rebuild train word2vec.=================')
            train_x, train_y = cls.load_news_train()
            wv_model = word2vec.Word2Vec(train_x, window=5, min_count=3, workers=4)
            print('===============rebuild train word2vec complete.=================')
            wv_model.save(wv_model_path)
            print('===============rebuild save word2vec complete.=================')
            return wv_model
        else:
            if os.path.exists(wv_model_path):
                print('===============load word2vec complete.=================')
                return models.Word2Vec.load(wv_model_path)
            else:
                print('===============begin train word2vec.=================')
                train_x, train_y = cls.load_news_train()
                wv_model = word2vec.Word2Vec(train_x, window=5, min_count=3, workers=4)
                print('===============train word2vec complete.=================')
                wv_model.save(wv_model_path)
                print('===============save word2vec complete.=================')
                return wv_model


if __name__ == '__main__':
    v = LoadCorpus.load_wv_model()
    for word in v.wv.vocab:
        print(v[word], word)
