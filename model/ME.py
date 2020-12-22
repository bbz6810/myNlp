import time
import numpy as np
from collections import defaultdict

from corpus.load_corpus import LoadCorpus


class MaxEnt:
    def __init__(self, train_x, train_y, test_x, test_y):
        self.train_x = train_x
        self.train_y = train_y
        self.test_x = test_x
        self.test_y = test_y
        self.feature_num = train_x.shape[1]

        self.N = train_x.shape[0]
        self.n = 0
        self.M = 10000
        self.fixy = self.calc_fixy()
        self.w = np.zeros(shape=(self.n,))
        self.xy2id_dict, self.id2xy_dict = self.create_search_dict()
        self.ep_xy = self.calc_epxy()

    def calc_fixy(self):
        '''
        计算xy在训练集中出现的次数
        :return:
        '''
        fixy_dict = [defaultdict(int) for i in range(self.feature_num)]
        for i in range(self.train_x.shape[0]):
            for j in range(self.feature_num):
                fixy_dict[j][(self.train_x[i][j], self.train_y[i])] += 1
        for i in fixy_dict:
            self.n += len(i)
        return fixy_dict

    def create_search_dict(self):
        xy2id_dict = [{} for i in range(self.feature_num)]
        id2xy_dict = {}
        index = 0
        for feature in range(self.feature_num):
            for (x, y) in self.fixy[feature]:
                xy2id_dict[feature][(x, y)] = index
                id2xy_dict[index] = (x, y)
                index += 1
        return xy2id_dict, id2xy_dict

    def calc_epxy(self):
        '''

        :return:
        '''
        ep_xy = [0] * self.n

        for feature in range(self.feature_num):
            for (x, y) in self.fixy[feature]:
                id = self.xy2id_dict[feature][(x, y)]
                ep_xy[id] = self.fixy[feature][(x, y)] / self.N
        return ep_xy

    # def calcepxy(self):
    #     epxy = [0] * self.n
    #     for i in range(self.N):
    #         pwxy = [0] * 2
    #         pwxy[0] = self.calcpwy_x(self.train_x[i], 0)
    #         pwxy[1] = self.calcpwy_x(self.train_x[i], 1)
    #
    #         for feature in range(self.feature_num):
    #             for y in range(2):
    #                 if (self.train_x[i][feature], y) in self.fixy[feature]:
    #                     id = self.xy2id_dict[feature][(self.train_x[i][feature], y)]
    #                     epxy[id] += (1 / self.N) * pwxy[y]
    #     return epxy
    def calcepxy(self):
        epxy = [0] * self.n
        for i in range(self.N):
            pwxy = [0] * 10
            for j in range(10):
                pwxy[j] = self.calcpwy_x(self.train_x[i], j)

            for feature in range(self.feature_num):
                for y in range(10):
                    if (self.train_x[i][feature], y) in self.fixy[feature]:
                        id = self.xy2id_dict[feature][(self.train_x[i][feature], y)]
                        epxy[id] += (1 / self.N) * pwxy[y]
        return epxy

    # def calcpwy_x(self, x, y):
    #     numerator = 0
    #     z = 0
    #     for i in range(self.feature_num):
    #         if (x[i], y) in self.xy2id_dict[i]:
    #             index = self.xy2id_dict[i][(x[i], y)]
    #             numerator += self.w[index]
    #         if (x[i], 1 - y) in self.xy2id_dict[i]:
    #             index = self.xy2id_dict[i][(x[i], 1 - y)]
    #             z += self.w[index]
    #
    #     numerator = np.exp(numerator)
    #     z = np.exp(z) + numerator
    #     return numerator / z
    def calcpwy_x(self, x, y):
        numerator = 0
        z = np.zeros(shape=(10,))
        for i in range(self.feature_num):
            for j in range(10):
                if (x[i], y) in self.xy2id_dict[i]:
                    index = self.xy2id_dict[i][(x[i], y)]
                    # 如果标签是该标签
                    if int(y) == int(j):
                        numerator += self.w[index]
                    else:
                        z[j] += self.w[index]
        numerator = np.exp(numerator)
        z = np.sum(np.exp(z))
        return numerator / z

    def max_entropy_train(self, ite=500):
        for i in range(ite):
            iterstart = time.time()
            epxy = self.calcepxy()

            # 使用IIS，设置sigma列表
            sigma_list = [0] * self.n
            for j in range(self.n):
                sigma_list[j] = (1 / self.M) * np.log(self.ep_xy[j] / epxy[j])

            self.w = [self.w[i] + sigma_list[i] for i in range(self.n)]

            # 单次迭代结束
            iterend = time.time()
            print('iter:{}-{}, time:{}'.format(i, ite, iterend - iterstart))
            score = self.test()
            print('score:{}'.format(score))

    def predict(self, x):
        result = [0] * 2
        for i in range(2):
            result[i] = self.calcpwy_x(x, i)
        return result.index(max(result))

    def test(self):
        error_count = 0
        for i in range(self.test_x.shape[0]):
            result = self.predict(self.test_x[i])
            if result != self.test_y[i]:
                error_count += 1
        return 1 - error_count / self.test_x.shape[0]


if __name__ == '__main__':
    train_x, train_y, test_x, test_y = LoadCorpus.load_mnist()
    me = MaxEnt(train_x, train_y, test_x, test_y)
    me.max_entropy_train()
