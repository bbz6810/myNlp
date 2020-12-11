import numpy as np
import math
import random

from corpus.load_corpus import LoadCorpus


class SVM:
    def __init__(self, train_x, train_y, sigma=10, c=200, toler=0.001):
        self.train_x = np.mat(train_x)
        self.train_y = np.mat(train_y).T
        self.m, self.n = np.shape(train_x)
        self.sigma = sigma
        self.c = c
        self.toler = toler

        self.k = self.calc_kernel()
        self.b = 0
        self.alpha = np.zeros(shape=(train_x.shape[0],))
        self.e = [0 * self.train_y[i, 0] for i in range(train_y.shape[0])]
        self.support_vec_index = []

    def calc_kernel(self):
        k = np.zeros(shape=(self.m, self.m))
        for i in range(self.m):
            x = self.train_x[i, :]
            for j in range(i, self.m):
                z = self.train_x[j, :]
                result = (x - z) * (x - z).T
                result = np.exp(-1 * result / (2 * self.sigma ** 2))
                k[i][j] = result
                k[j][i] = result
        return k

    def calc_gxi(self, i):
        gxi = 0
        index = [i for i, alpha in enumerate(self.alpha) if alpha != 0]
        for j in index:
            gxi += self.alpha[j] * self.train_y[j] * self.k[j][i]
        gxi += self.b
        return gxi

    def is_satisfy_kkt(self, i):
        gxi = self.calc_gxi(i)
        yi = self.train_y[i]
        if (math.fabs(self.alpha[i]) < self.toler) and (yi * gxi >= 1):
            return True
        elif (math.fabs(self.alpha[i] - self.c) < self.toler) and (yi * gxi <= 1):
            return True
        elif (self.alpha[i] > -self.toler) and (self.alpha[i] < (self.c + self.toler)) and (
                math.fabs(yi * gxi - 1) < self.toler):
            return True
        return False

    def calc_ei(self, i):
        gxi = self.calc_gxi(i)
        return gxi - self.train_y[i]

    def get_alpha_j(self, e1, i):
        e2 = 0
        max_e1_e2 = -1
        max_index = -1
        nozero_e = [i for i, ei in enumerate(self.e) if ei != 0]
        for j in nozero_e:
            e2_tmp = self.calc_ei(j)
            if math.fabs(e1 - e2_tmp) > max_e1_e2:
                max_e1_e2 = math.fabs(e1 - e2_tmp)
                e2 = e2_tmp
                max_index = j
        if max_index == -1:
            max_index = i
            while max_index == i:
                max_index = int(random.uniform(0, self.m))
            e2 = self.calc_ei(max_index)
        return e2, max_index

    def calc_singl_kernel(self, x1, x2):
        result = (x1 - x2) * (x1 - x2).T
        result = np.exp(-1 * result / (2 * self.alpha ** 2))
        return np.exp(result)

    def train(self, ite=100):
        iter_setp = 0
        parameter_changed = 1
        while (iter_setp < ite) and parameter_changed > 0:
            print('iter:{}'.format(iter_setp))
            iter_setp += 1
            parameter_changed = 0
            for i in range(self.m):
                if self.is_satisfy_kkt(i) is False:
                    e1 = self.calc_ei(i)
                    e2, j = self.get_alpha_j(e1, i)

                    y1 = self.train_y[i]
                    y2 = self.train_y[j]

                    alpha_old1 = self.alpha[i]
                    alpha_old2 = self.alpha[j]
                    if y1 != y2:
                        l = max(0, alpha_old2 - alpha_old1)
                        h = min(self.c, self.c + alpha_old2 - alpha_old1)
                    else:
                        l = max(0, alpha_old2 + alpha_old1 - self.c)
                        h = min(self.c, alpha_old2 + alpha_old1)
                    if l == h:
                        continue

                    k11 = self.k[i][i]
                    k22 = self.k[j][j]
                    k21 = self.k[j][i]
                    k12 = self.k[i][j]

                    alpha_new2 = alpha_old2 + y2 * (e1 - e2) / (k11 + k22 - 2 * k12)
                    if alpha_new2 < l:
                        alpha_new2 = l
                    elif alpha_new2 > h:
                        alpha_new2 = h
                    alpha_new1 = alpha_old1 + y1 * y2 * (alpha_old2 - alpha_new2)

                    b1_new = -1 * e1 - y1 * k11 * (alpha_new1 - alpha_old1) - y2 * k21 * (
                            alpha_new2 - alpha_old2) + self.b
                    b2_new = -1 * e2 - y1 * k12 * (alpha_new1 - alpha_old1) - y2 * k22 * (
                            alpha_new2 - alpha_old2) + self.b

                    if (alpha_new1 > 0) and (alpha_new1 < self.c):
                        b_new = b1_new
                    elif (alpha_new2 > 0) and (alpha_new2 < self.c):
                        b_new = b2_new
                    else:
                        b_new = (b1_new + b2_new) / 2

                    self.alpha[i] = alpha_new1
                    self.alpha[j] = alpha_new2
                    self.b = b_new

                    self.e[i] = self.calc_ei(i)
                    self.e[j] = self.calc_ei(j)

                    if math.fabs(alpha_new2 - alpha_old2) >= 0.00001:
                        parameter_changed += 1
            print('iter:{}'.format(iter_setp))

        for i in range(self.m):
            if self.alpha[i] > 0:
                self.support_vec_index.append(i)

    def predict(self, x):
        result = 0
        for i in self.support_vec_index:
            tmp = self.calc_singl_kernel(self.train_x[i, :], np.mat(x))
            result += self.alpha[i] * self.train_y[i] * tmp
        result += self.b
        return np.sign(result)

    def test(self, test_x, test_y):
        error_cnt = 0
        for i in range(test_x.shape[0]):
            result = self.predict(test_x[i])
            if result != test_y[i]:
                error_cnt += 1
        return 1 - error_cnt / test_x.shape[0]


if __name__ == '__main__':
    train_x, train_y, test_x, test_y = LoadCorpus.load_mnist()
    svm = SVM(train_x, train_y, 10, 200, 0.001)
    svm.train()
    score = svm.test(test_x, test_y)
    print('score', score)
