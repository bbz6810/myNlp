"""
    ent = ∑ p * log(p)
    p: 一个事件发生的概率
"""

import math


class Ent:
    def __init__(self):
        pass

    def calc_ent(self, data):
        # data = [
        #     [[1, 1, 1], 1],
        #     [[0, 0, 0], 0]
        # ]

        d = {}
        for t in data:
            d[t[1]] = d.setdefault(t[1], 0) + 1
        d_total = sum(d.values())
        ent = 0
        for key, value in d.items():
            ent += math.log(value / d_total, 2)
        return -ent


if __name__ == '__main__':
    print(Ent().calc_ent(''))
