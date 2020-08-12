"""决策树

    1、选择最优属性
    2、根据最优属性划分数据集，递归建树

        1 1 1  yes
        1 1 0  yes
        1 0 1  no
        1 0 0  no

"""

import copy
import time

from model.feature_select.gain import Gain


class TreeNode:
    def __init__(self):
        self.attr = None
        self.attr_data = {}
        self.attr_dict = {}
        self.leaf = False
        self.c = None

    def __str__(self):
        return 'TreeNode: \n' \
               '当前节点属性:{}\n' \
               '当前节点状态:{}\n' \
               '当前节点分类:{}\n'.format(self.attr, self.leaf, self.c)


gain = Gain()


def split_data_list(data_list, attr_index, value):
    """根据属性索引删除该列数据，并且返回值等于value的集合

    :param data_list:
    :param attr_index:
    :param value:
    :return:
    """
    new_data = []
    for data in data_list:
        if data[0][attr_index] == value:
            tmp = list(data[0][:attr_index])
            tmp.extend(data[0][attr_index + 1:])
            # tmp = list(data[0])
            new_data.append([tmp, data[1]])
    return new_data


def create_tree(data_list, attr_index_list, feature_label):
    """

    :param data_list: 训练数据集
    :param attr_list: 可供选择的子属性集
    :return:
    """
    labels = [data[1] for data in data_list]
    if labels.count(labels[0]) == len(labels):
        return labels[0]

    if len(data_list[0]) == 1:
        vote = dict()
        for label in labels:
            vote[label] = vote.get(label, 0) + 1
        return sorted(vote.items(), key=lambda x: x[1], reverse=True)[0][0]

    best_attr_index = gain.calc_gain_all(data_list, attr_index_list)
    best_attr = feature_label[best_attr_index]
    # print("最好的属性索引", best_attr_index, len(data_list))

    my_tree = {best_attr: {}}
    attr_index_list.append(best_attr_index)
    attr_values = set([i[0][best_attr_index] for i in data_list])
    # print('该属性值', attr_values)

    # print('当前树结构', my_tree)
    # time.sleep(3)
    d = 0
    for value in attr_values:
        d += 1
        sub_attr_index_list = attr_index_list[:]
        tmp = split_data_list(data_list, best_attr_index, value)
        # print('属性遍历-------->>>>>', value, tmp, sub_attr_index_list)
        my_tree[best_attr][value] = create_tree(tmp, sub_attr_index_list, feature_label)
        # print('my_tree', my_tree)
    return my_tree



if __name__ == '__main__':
    tree = TreeNode()
