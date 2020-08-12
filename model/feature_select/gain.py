""" ID3算法
    gain: ent(C) - ∑ ((D_i/D) * ent(C_i))

"""

from model.feature_select.ent import Ent


class Gain(Ent):
    def __init__(self):
        super().__init__()

    def calc_gain_by_attr(self, data, attr_index):
        ent = self.calc_ent(data)
        one_attr_dict = dict()
        for row in data:
            if row[0][attr_index] not in one_attr_dict:
                one_attr_dict[row[0][attr_index]] = []
            one_attr_dict[row[0][attr_index]].append(row)
        tmp = 0
        for key, value in one_attr_dict.items():
            tmp += (len(value) / len(data)) * self.calc_ent(value)
        return ent - tmp

    def calc_gain_all(self, data, used_attr_list):
        max_attr = -1
        max_value = 0
        for i in range(len(data[0][0])):
            if i in used_attr_list:
                continue
            tmp = self.calc_gain_by_attr(data, i)
            if tmp > max_value:
                max_value = tmp
                max_attr = i
        return max_attr
