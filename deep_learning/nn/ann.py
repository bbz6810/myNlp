"""人工神经网络ANN---最基本的网络结构
    误差逆传播算法：https://blog.csdn.net/qq_32241189/article/details/80305566
    误差逆传播算法：https://www.cnblogs.com/liuwu265/p/4696388.html

    组成：
        1、网络架构
        2、激活函数
        3、找出最优权重值的参数学习算法(BP算法)

    过程：
        1、将网络中的所有权值初始化
        2、对每个样例执行前向算法
        3、执行BP算法误差回传
            a、输出层误差逆传播
            b、隐藏层误差逆传播-间接求导计算

    w:参数
    b:参数
    o = w * x + b     对w求导得 x， 对b求导得 1
    out = f(o)        对f求导得 f' = f * (1-f) [f为sigmoid函数]
    损失函数L：求和 0.5 * (y - out)^2   对L_w求导得 (y - out) * f * (1-f) * x
                                      对L_b求导得 (y - out) * f * (1-f)


    二层网络求导过程
    E = 0.5 * SUM[(真实值-预测值)^2]

    E/w_ij = E/y_j * y_j/w_ij
    E/y_j = (真实值-预测值)
    y_j/w_ij = y_j/u_j * u_j/w_ij

    E/w_ij = (真实值-预测值) * y_j(1-y_j) * x_i

    三层网络求导过程
    1、隐藏层到输出层
        E/w_2jk = E/y_k * y_k/u_2k * u_2k/w_2jk
                = (真实值-预测值) * y_k(1-y_k) * x_j
    2、输出层到隐藏层
        E/w_1ij = SUM[E/y_k * y_k/u_2k * u_2k/w_1ij]

        u_2k/w_1ij = u_2k/z_j * z_j/w_1ij = w_2jk * z_j/w_1ij

        z_j/w_1ij = z_j/u_1j * u_1j/w_1ij = z_j(1-z_j) * x_i

        E/w_1ij = sum(E/y_k * y_k/u_2k * u_2k/z_j * z_j/u_1j * u_1j/w_1ij)
                = sum((真实值-预测值) * y_k(1-y_k) * w_2jk * z_j(1-z_j) * x_i)


"""

import numpy as np

from corpus import line_r_path
from tools.plot import display


def load_line_r_data():
    t = []
    with open(line_r_path, encoding='utf8', mode='r') as f:
        for line in f.readlines()[1:]:
            s = line.strip().split(',')
            if s[0] != '多' and s[1] != '多' and s[2] != '多' and s[-1] != '多':
                t.append([s[0], s[1], s[2], s[-1]])

    data = np.array(t, dtype='float32')
    x = np.array(data[:, :3])
    y = np.array(data[:, 3:])
    return x, y


def generate_data():
    x = np.linspace(-2, 2, 20)[np.newaxis, :].T
    noise = np.random.normal(0.0, 0.5, size=(1, 20)).T
    y = x ** 2 + noise
    return x, y


class ANN1:
    def __init__(self):
        self.input = None
        self.y = None
        self.weight1 = None
        self.weight2 = None
        self.output = None

        self.z1 = None
        self.a1 = None
        self.z2 = None
        self.a2 = None

    def init_param(self, x, y):
        self.input = x
        self.y = y

        self.weight1 = np.random.rand(self.input.shape[1], 4)
        self.weight2 = np.random.rand(4, 1)

        self.output = np.zeros(self.y.shape)

    def fp(self):
        self.z1 = np.dot(self.input, self.weight1)
        self.a1 = 1 / (1 + np.exp(-self.z1))
        print('------------>', self.a1)

        self.z2 = np.dot(self.a1, self.weight2)
        # self.a2 = 1 / (1 + np.exp(-self.z2))
        self.a2 = self.z2

    def bp(self):
        w2 = np.dot(self.a1.T, (self.y - self.a2) * (self.a2 * (1 - self.a2)))
        w1 = np.dot(self.input.T,
                    (np.dot((self.y - self.a2) * (self.a2 * (1 - self.a2)), self.weight2.T) * self.a1 * (1 - self.a1)))
        self.weight1 += w1
        self.weight2 += w2

    def train(self, x, y):
        self.init_param(x, y)
        epho = 10000
        loss = np.zeros(epho)
        for i in range(epho):
            self.fp()
            self.bp()
            loss[i] = np.sum((self.a2 - y) ** 2)
        display(loss)


class ANN2:
    def __init__(self):
        pass


if __name__ == '__main__':
    # x, y = load_line_r_data()
    # x = np.array([[0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 1]])
    # y = np.array([[0, 1, 1, 0]]).T
    x, y = generate_data()
    print(x[:2])
    print(y[:2])
    ann = ANN1()
    ann.train(x, y)
