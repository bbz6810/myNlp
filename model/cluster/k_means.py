"""k_means聚类算法
    1、数据的距离度量公式d的定义
    2、随机选择K个点作为聚类中心
    3、计算所有点与哪个点近就归为哪一类
    4、一个轮次之后更新聚类中心
    5、重复3,4直到聚类中心不再变化

    [
        输入：样本集D={x1, x2, x3.....xm}
             聚类簇数k
        过程：
            从D中随机选择k个样本作为初始均值向量{m1, m2,...mk}
            repeat
                令C_i = null (1 <= i <= k)
                for j = 1,2,...m do
                    计算样本x_j与各均值向量m的距离：d_ji=||x_j-m_i||_2
                    根据距离最近的均值向量确定x_j的簇标记
                    将样本x_j划入相应的簇
                end for
                for i=1,2....k do
                    计算新均值向量并更新
            until 当前均值向量均未更新

        输出：C = {c_1, c_2....c_k}
    ]

"""
from copy import deepcopy
import numpy as np

from tools.distance import euclidean_distance

MAX_DISTANCE = float('inf')

x1 = [1, 2, 3, 1, 5, 6, 5, 5, 6, 7, 8, 9, 7, 9]  # x坐标列表
x2 = [1, 3, 2, 2, 8, 6, 7, 6, 7, 1, 2, 1, 1, 3]  # y坐标列表


def gen_data_set(x, y):
    t = [[i, j] for i, j in zip(x, y)]
    return t
    # return np.array(t, dtype='float32')


class KMeans:
    def __init__(self, k):
        self.k = k
        self.k_array = None

    def calc_k_vector(self, c_list):
        tmp = deepcopy(self.k_array)
        for index, value in enumerate(c_list):
            self.k_array[index] = np.array(sum(value) / len(value))
        return not (tmp == self.k_array).all()

    def cluster(self, x):
        np.random.shuffle(x)
        self.k_array = x[:3]
        # self.k_array = np.array([[9., 1.], [5., 8.], [1., 1.]])
        while True:
            tmp_c_list = []
            for i in range(self.k):
                tmp_c_list.append([])
            # 计算每个样本与每个簇的距离
            for i in range(x.shape[0]):
                min_distance, index = MAX_DISTANCE, self.k + 1
                for m in range(self.k_array.shape[0]):
                    i_m_distance = euclidean_distance(x[i], self.k_array[m])
                    if i_m_distance < min_distance:
                        min_distance, index = i_m_distance, m
                tmp_c_list[index].append(x[i])
            cluster_change = self.calc_k_vector(tmp_c_list)
            print(cluster_change)
            if not cluster_change:
                break


if __name__ == '__main__':
    x = gen_data_set(x1, x2)
    k_means = KMeans(3)
    k_means.cluster(x)
