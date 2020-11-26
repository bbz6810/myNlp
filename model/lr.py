"""
    逻辑回归计算步骤总结
    w (1 * n)  x = (n * 1)
    1、确定一个样本的概率
        假设只有两个标签
        p = 1 / (1 + e ^ -wx)

        P_y=1 = 1 / (1+e ^ -wx)
        p_y=0 = e ^ -wx / (1+ e ^ -wx)

        每个样本发生的概率（y_i = 1 则为p，y_i = 0 则为 1-p ）
        P(y_i / x_i) = p ^ y_i * (1-p) ^ 1-y_i

    2、计算所有样本的发生概率
        P(总) = P(y_1 / x_1) * P(y_2 / x_2) * ... * P(y_n / x_n)
              = ∏ p ^ y_i * (1-p) ^ 1-y_i
        注：只有一个变量w，在p里面

    3、连乘变连加：由于连乘太复杂容易发生溢出，固对连乘取ln变成连加
        F(w) = ln(P(总))
             = ln( ∏ p ^ y_i * (1-p) ^ 1-y_i)
             = ∑ ln( p ^ y_i * (1-p) ^ 1-y_i)
             = ∑ ( y_i * ln(p) + (1-y_i) * ln(1-p) )

        即为损失函数：求损失函数的最小值可在其前加个负号

    4、最大似然估计：
        w * = arg max_w F(w) = - arg min_w F(w)
        只需求出w*使-F(w)最小
        由于F(w)是连续的凸函数，所以容易求出极值

    5、求F(w)的梯度值
        1.求p的导数
            p' = f'(w) = (1 / (1+e^-wx))'
               = - (1/ (1+e^-wx)^2) * (e^-wx)'
               = - (1/ (1+e^-wx)^2) * e^-wx * (-wx)'
               = - (1/ (1+e^-wx)^2) * e^-wx * -x
               = - (e^-wx / (1+e^-wx)^2) * -x
               = (e^-wx / (1+e^-wx)^2) * x
               = (1 / 1 + e^-wx) * (e^-wx / 1 + e^-wx) * x
               = p * (1-p) * x
            同理：(1-p)' = -p(1-p)x

        2.把p的导数带入F(w)中求梯度
            F(w) = (∑(y_i * ln(p) + (1-y_i) * ln(1-p)))'
                 = ∑ [(y_i * ln'(p) + (1-y_i) * ln'(1-p))]
                 = ∑ [(y_i * (1/p) * p') + (1-y_i) * (1/(1-p)) * (1-p)']
                 = ∑ [(y_i * (1/p) * p*(1-p)*x_i) + (1-y_i) * (1/(1-p)) * (-p*(1-p)*x_i)]
                 = ∑ [(y_i * (1-p) * x_i) + (1-y_i) * (-p * x_i)]
                 = ∑ [(y_i * (1-p) * x_i) - (1-y_i) * (p * x_i)]
                 = ∑ [(y_i * x - y_i * x_i * p) - (p * x_i - p * x_i * y_i)]
                 = ∑ [y_i * x - y_i * x_i * p - p * x_i + p * x_i * y_i]
                 = ∑ [y_i * x_i - p * x_i]
                 = ∑ (y_i - p) * x_i

                 = ∑ (y_i - (1 / (1+e^-wx_i))) * x_i

    6、梯度下降GD和随机梯度下降SGD
        梯度下降法：
            w_t+1 = w_t + η F(w)
"""

import numpy as np
from sklearn.datasets import load_iris
from tools import sigmoid
from corpus.load_corpus import LoadCorpus
from corpus import mnist_x_train_path, mnist_y_train_path, mnist_x_test_path, mnist_y_test_path


def load_data(fname='ytb_lr.txt'):
    x = []
    y = []
    with open(fname, encoding='utf8', mode='r') as f:
        for line in f.readlines():
            line = line.strip().split('\t')
            x.append([float(line[0]), float(line[1])])
            y.append(int(line[2]))
    return x, y


class LR:
    def __init__(self):
        self.w = None

    def save(self):
        pass

    def load(self):
        pass

    def add_b(self, x, y):
        x = np.mat(x)
        y = np.mat(y).transpose()
        b = np.ones((x.shape[0], 1))
        x = np.column_stack((b, x))
        return x, y

    def train(self, x_train, y_train):
        w = np.ones((x_train.shape[1], 1))
        learning_rate = 0.001
        for i in range(100000):
            h = sigmoid(np.dot(x_train, w))
            error = y_train - h
            w = w + learning_rate * np.dot(x_train.transpose(), error)
        self.w = w
        print('系数w', self.w, self.w.shape)

    def predict(self, x):
        wx = np.dot(x, self.w)
        return sigmoid(wx)


class LogicR:
    def __init__(self):
        self.data = load_iris()

    def load_data(self):
        # location = np.random.permutation(100)
        # return self.data.data[:100][location][:80], self.data.target[:100][location][:80], \
        #        self.data.data[:100][location][80:], self.data.target[:100][location][80:]
        x, y, x_test, y_test = LoadCorpus.load_mnist(mnist_x_train_path, mnist_y_train_path, mnist_x_test_path,
                                                     mnist_y_test_path)
        # print(x.shape)
        # print(y.shape)
        # print(x_test.shape)
        # print(y_test.shape)
        return x, y, x_test, y_test

    def logistic_regression(self, x, y, iters=200):
        one = np.ones(shape=(len(x), 1))
        x = np.column_stack((x, one))
        for idx, i in enumerate(y):
            if i == 2:
                y[idx] = 0
        w = np.zeros(x.shape[1])

        h = 0.001

        for i in range(iters):
            for j in range(x.shape[0]):
                wx = np.dot(w, x[j])
                yi = y[j]
                xi = x[j]
                w += h * (xi * yi - (np.exp(wx) * xi) / (1 + np.exp(wx)))
        return w

    def predict(self, x_i, w):
        wx = np.dot(w, x_i)
        p1 = np.exp(wx) / (1 + np.exp(wx))
        return 1 if p1 >= 0.5 else 0

    def test(self, x, y, w):
        one = np.ones(shape=(x.shape[0], 1))
        x = np.column_stack((x, one))

        error_count = 0
        for i in range(x.shape[0]):
            if y[i] != self.predict(x[i], w):
                error_count += 1
        return 1 - error_count / x.shape[0]


if __name__ == '__main__':
    lr = LogicR()
    x, y, testx, testy = lr.load_data()
    w = lr.logistic_regression(x, y)
    score = lr.test(testx, testy, w)
    print(score)
    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression()
    lr.fit(x, y)
    s = lr.score(testx, testy)
    print(s)
