"""sim hash算法
    1、分词
    2、去除停用词
    3、形成词和w的对
    4、hash词并权重相乘
    5、所有词相加
    6、降维>0 1 <0 0
    7、计算海明距离 异或
    8、小于3的一般是相似文档

    优化策略：
        把64位的hash值拆成4份，一份16位
        如果汉明距离设置成3，则肯定有一个会出现在这4份数据之中
        同理可扩展至多位

"""

import jieba

from tools import filter_stop, hash2bin, hamming


class SimHash:
    def __init__(self):
        pass

    def save(self):
        pass

    def load(self):
        pass

    def train(self):
        pass

    def predict(self):
        pass

    def merge_word(self, d):
        tmp = [0] * 64
        for k, v in d.items():
            k_bin = hash2bin(k)
            for idx, c in enumerate(k_bin):
                if c == '1':
                    tmp[idx] += v
                else:
                    tmp[idx] -= v
        t = ''
        for c in tmp:
            if c > 0:
                t += '1'
            else:
                t += '0'
        return int(t, base=2)

    def simhash(self, doc1, doc2):
        d1 = filter_stop(jieba.cut(doc1))
        d2 = filter_stop(jieba.cut(doc2))
        d1_dict = {}
        d2_dict = {}
        for word in d1:
            if word in d1_dict:
                d1_dict[word] += 1
            else:
                d1_dict[word] = 1
        for word in d2:
            if word in d2_dict:
                d2_dict[word] += 1
            else:
                d2_dict[word] = 1
        n1 = self.merge_word(d1_dict)
        n2 = self.merge_word(d2_dict)
        t = hamming(n1, n2)
        print(t)


if __name__ == '__main__':
    s1 = ''
    s2 = ''
    SimHash().simhash(s1, s2)
