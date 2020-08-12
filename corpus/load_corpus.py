import re
import jieba
import random

from gensim.models import KeyedVectors

from corpus import wv_model_path, news_jieba_path, chatbot100_path
from tools import n_gram, running_of_time


class LoadCorpus:
    def __init__(self):
        pass

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
    def load_wv_model(cls):
        """加载人民日报语料生成的word2vec模型

        :return:
        """
        return KeyedVectors.load_word2vec_format(wv_model_path)


if __name__ == '__main__':
    LoadCorpus.load_chatbot100_train()
