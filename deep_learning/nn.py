import numpy as np


class NN:
    def __init__(self, nn_param):
        self.nn = None
        self.nn_param = nn_param

    def score(self, _y, y):
        if self.nn_param.class_num == 2:
            print("正确个数", np.sum((_y > 0.5 + 0) == y), '总数', y.shape[0], '正确率',
                  np.sum((_y > 0.5 + 0) == y) / y.shape[0])
        else:
            right = 0
            for index, _i in enumerate(_y):
                if y[index][_i.argmax()] == 1:
                    right += 1
            print('正确个数', right, '总数', y.shape[0], '正确率', right / y.shape[0])
