"""线性回归

    矩阵求导公式：
        a是实数，β，X是向量， A, B, C是与 X 无关的矩阵
        对X求导  d(β^T * X) / dX = β
                d(X^T * X) / dX = 2X   X^2' = 2X
                d(X^T * AX) / dX = (A + A^T) * X

                dX^T / dX = I
                dX / dX^T = I
                dX^T * A / dX = A
                dA * X / dX^T = A
                dX * A / dX = A^T
                dA * X / dX = A^T

                dU / dX^T = (dU^T/dX)^T
                dU^T * V / dX = (dU^T/dX)*V + (dV^T/dX)*U^T

                dAB / dX = (dA / dX)*B + A*(dB/dX)
                dU^T * XV / dX = UV^T

    损失函数：
            L1：|y_i - y|
            L2: 求和 1/2n (y_1 - y)^2

            残差平方和：点到直线的平行于y轴的距离
            均方误差：残差平方和除以样本数 《作为线性回归的损失函数》== 最小二乘法
    假设损失函数服从高斯分布：
        (1 / ( (σ √2π)) ) * exp ^ -((x-μ)^2 / 2σ^2)

    两种方法求解
    1、解析解：根据损失函数直接求W    (X^T * X)^-1 * X^T * Y
    2、优化解：利用梯度下降求W
        损失函数: y = 1/2m 求和 (W^TX - Y)^2
        递推公式: w = w - a * 1/m * X^T* (W^TX - Y)

"""

import os

import numpy as np
from corpus import line_r_path


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


class LinerR:
    def __init__(self, gd=True):
        self.w = None
        self.gd = gd

    def w_(self, x, y):
        return np.dot(np.linalg.inv(np.dot(x.T, x)), np.dot(x.T, y))

    def wdotx(self, w, x):
        return np.dot(x, w)

    def cost(self, w, x, y):
        error = self.wdotx(w, x) - y
        # print(error)
        item = error ** 2
        return np.sum(item) / (2 * x.shape[0])

    def gradient(self, w, x, y):
        """1/m * X^T* (W^TX - Y)

        :return:
        """
        return np.dot(x.T, self.wdotx(w, x) - y) / x.shape[0]

    def stop_strage(self, cost_pre, cost_current, min_t):
        return (cost_pre >= cost_current) and (cost_pre - cost_current < min_t)

    def train(self):
        x, y = load_line_r_data()

        x = np.array([[1, 1], [2, 2], [3, 3]])
        y = np.array([[1], [2], [3]])
        _one = np.ones(shape=(x.shape[0], 1))
        x = np.column_stack([_one, x])
        w = np.random.rand(x.shape[1], 1)
        print(w)

        # l = LinearRegression()
        # l.fit(x, y)
        # print(l.coef_)
        # return

        if self.gd is False:
            self.w = self.w_(x, y)
            print(self.w)
            """
            [[-76.78838164]
            [ 27.91364962]
            [ 76.50599937]
            [  6.18406516]]
            """
        else:
            a = 0.001
            cost_val = self.cost(w, x, y)
            # print(cost_val)
            # return
            for i in range(20000):
                try:
                    w = w - a * self.gradient(w, x, y)
                    update_cost = self.cost(w, x, y)
                except Exception:
                    break
            self.w = w
            print('w ===', self.w)

    def predict(self):
        x = np.array([[1, 4, 4]], dtype='float32')
        y = np.dot(x, self.w)
        print(y)


if __name__ == '__main__':
    liner = LinerR(True)
    liner.train()
    liner.predict()
