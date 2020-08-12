"""db_scan
    1、邻域-核心对象-密度直达-密度可达-密度相连
    2、密度可达的所有点的集合为簇
    3、利用队列像树的层次遍历方法类似

    输入：样本集D，邻域参数（e，min_pts）

    过程：
        初始化核心对象集合：o
        for j = 1,2,,,m do
            确定样本xj的邻域 N（j）
            if N（j） > min_pts then
                将样本xj加入核心对象集合 o
            end if
        end for

        初始化聚类簇数k=0
        初始化未访问样本集合T
        while o != 空 do
            记录当前未访问样本集合：T_old = T
            随机选取一个核心对象o，初始化队列Q=<o>
            T = T 除去 {o}
            while Q ！= 空 do
                取出队列的首个样本q
                if N(q) >= min_pts then
                    将N（q）与T 的交集加入队列Q
                    T = T 除去 N（q）与T的交集
                end if
            end while
            k = k + 1 生成聚类簇Ck = T_old 除去T
            o = o 除去 Ck
        end while

"""
import queue
from copy import deepcopy
import numpy as np

from tools.distance import euclidean_distance
from model.cluster.k_means import gen_data_set

x1 = [1, 2, 3, 1, 5, 6, 5, 5, 6, 7, 8, 9, 7, 9]  # x坐标列表
x2 = [1, 3, 2, 2, 8, 6, 7, 6, 7, 1, 2, 1, 1, 3]  # y坐标列表


# x1 = [1, 2, 3]
# x2 = [1, 3, 2]


class DBScan:
    def __init__(self, min_distance, min_k):
        self.min_distance = min_distance
        self.k = min_k
        self.neighbourhood_list = list()  # [ [当前数据点, [当前数据点邻域]], [...]...       ]
        self.core_list = list()  # [ [当前数据点, [当前数据点核心对象]], [...]...       ]

    def d(self, c1, c2):
        total = 0
        for i in c1:
            for j in c2:
                total += euclidean_distance(i, j)
        return total / (len(c1) * len(c2))

    def neighbourhood(self, data_list):
        for index_i, data1 in enumerate(data_list):
            tmp = []
            for index_j, data2 in enumerate(data_list):
                if index_i == index_j:
                    continue
                dis = self.d([data1], [data2])
                if dis <= self.min_distance:
                    tmp.append(data2)
            self.neighbourhood_list.append([data1, tmp])

    def core_object(self):
        for index, value in enumerate(self.neighbourhood_list):
            if len(value[1]) >= self.k:
                self.core_list.append(value)

    def get_core_object(self, x):
        for i in self.core_list:
            if i[0] == x:
                return i[1]

    def del_value(self, value, data_list):
        for i in range(len(data_list)):
            if value == data_list[i]:
                tmp = list(data_list[:i])
                tmp.extend(data_list[i + 1:])
                return tmp

    def get_intersection(self, data_list, core_list):
        """取数据和核心对象的交集：如果数据里没有该核心对象说明该核心对象已经计算过

        :param data_list: 数据列表
        :param core_list: 核心对象列表
        :return:
        """
        retv_list = []
        for j in core_list:
            for i in data_list:
                if i == j:
                    retv_list.append(j)
                    break
        return retv_list

    def split_a_b(self, a, b):
        have_set = set()
        # print('a', a)
        # print('b', b)
        for i in b:
            # print('a', a)
            # print('i', i)
            t_list = self.find_T_k(a, i)
            have_set = have_set.union(set(t_list))

        c_list = []
        for index, j in enumerate(a):
            if index in have_set:
                continue
            else:
                c_list.append(j)
        return c_list

    def split_core_list_a_b(self, a, b):
        a_list = [i[0] for i in a]
        have_set = set()
        for i in b:
            t_list = self.find_T_k(a_list, i)
            have_set = have_set.union(set(t_list))

        c_list = []
        for index, j in enumerate(a_list):
            if index in have_set:
                continue
            else:
                c_list.append(a[index])
        return c_list

    def find_T_k(self, T, k):
        retv_list = []
        for index, i in enumerate(T):
            if k == i:
                retv_list.append(index)
        return retv_list

    def train(self, data_list):
        k = 0
        T = deepcopy(data_list)
        self.neighbourhood(data_list)
        # print('邻域', len(self.neighbourhood_list))
        # for i in self.neighbourhood_list:
        #     print(i[0], i[1], len(i[1]))
        self.core_object()
        # print('核心对象', len(self.core_list))
        # for i in self.core_list:
        #     print(i)
        c = []
        while self.core_list:
            # print('0000000', len(self.core_list), self.core_list)
            T_old = deepcopy(T)
            core_obj = self.core_list[0]
            q = queue.Queue()
            q.put(core_obj[0])
            T = self.del_value(core_obj[0], T)
            if T is None:
                break
            while not q.empty():
                q_first = q.get()
                # if q_first == [1, 1]:
                #     print('===============', q_first, type(q_first))
                core_list = self.get_core_object(q_first)
                # if q_first == [3, 2]:
                #     print('===============', core_list, T)
                # 如果队列里取的元素存在核心对象
                if core_list is not None:
                    core_value = self.get_intersection(T, core_list)
                    for one in core_value:
                        # 加入队列
                        q.put(one)
                        # 从T中删除该元素，说明该元素已经计算过
                        T = self.del_value(one, T)
                        # print('T', T, '-----------', one)
            k += 1
            # print('T', T, len(T))
            # print('T_old', T_old, len(T_old))
            have_list = self.split_a_b(T_old, T)
            # print('have list', have_list)
            c.append(have_list)
            # print(222222222222, len(self.core_list), self.core_list)
            # print(33333333333, have_list)
            self.core_list = self.split_core_list_a_b(self.core_list, have_list)
            # print(44444444, len(self.core_list), self.core_list)
            # break
        for i in c:
            print('类别', len(i), i)


if __name__ == '__main__':
    data_list = gen_data_set(x1, x2)
    # data_list = [[-1.26, 0.46],
    #              [-1.15, 0.49],
    #              [-1.19, 0.36],
    #              [-1.33, 0.28],
    #              [-1.06, 0.22],
    #              [-1.27, 0.03],
    #              [-1.28, 0.15],
    #              [-1.06, 0.08],
    #              [-1.00, 0.38],
    #              [-0.44, 0.29],
    #              [-0.37, 0.45],
    #              [-0.22, 0.36],
    #              [-0.34, 0.18],
    #              [-0.42, 0.06],
    #              [-0.11, 0.12],
    #              [-0.17, 0.32],
    #              [-0.27, 0.08],
    #              [-0.49, -0.34],
    #              [-0.39, -0.28],
    #              [-0.40, -0.45],
    #              [-0.15, -0.33],
    #              [-0.15, -0.21],
    #              [-0.33, -0.30],
    #              [-0.23, -0.45],
    #              [-0.27, -0.59],
    #              [-0.61, -0.65],
    #              [-0.61, -0.53],
    #              [-0.52, -0.53],
    #              [-0.42, -0.56],
    #              [-1.39, -0.26]]
    print('数据长度', len(data_list), data_list)

    db_scan = DBScan(2, 3)
    db_scan.train(data_list)

    from sklearn.cluster import DBSCAN

    d = DBSCAN(eps=2, min_samples=5)
    d.fit(data_list)
    print(d.labels_)

