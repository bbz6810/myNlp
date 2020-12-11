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
import numpy as np

from model.feature_select.gain import Gain
from corpus.load_corpus import LoadCorpus


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


def create_tree1(data_list, attr_index_list, feature_label):
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


'''
李航
'''


def major_class(y):
    class_dict = {}
    for i in range(len(y)):
        if y[i] in class_dict.keys():
            class_dict[y[i]] += 1
        else:
            class_dict[y[i]] = 1
    class_sort = sorted(class_dict.items(), key=lambda x: x[1], reverse=True)
    return class_sort[0][0]


def calc_h_d(y):
    h_d = 0
    train_label_set = set([label for label in y])
    for i in train_label_set:
        p = y[y == i].size / y.size
        h_d += -1 * p * np.log2(p)
    return h_d


def calc_h_d_a(train_attr_dev_feature, y):
    h_d_a = 0
    train_data_set = set([label for label in train_attr_dev_feature])
    for i in train_data_set:
        h_d_a += train_attr_dev_feature[train_attr_dev_feature == i].size / train_attr_dev_feature.size * calc_h_d(
            y[train_attr_dev_feature == i])
    return h_d_a


def calc_best_feature(train_x, train_y):
    feature_num = train_x.shape[1]
    max_g_d_a = -1
    max_feature = -1
    h_d = calc_h_d(train_y)
    for feature in range(feature_num):
        train_data_attr_devide_feature = np.array(train_x[:, feature].flat)
        g_d_a = h_d - calc_h_d_a(train_data_attr_devide_feature, train_y)
        if g_d_a > max_g_d_a:
            max_g_d_a = g_d_a
            max_feature = feature
    return max_feature, max_g_d_a


def get_sub_data_attr(train_data_attr, train_label_attr, A, a):
    print('1111', train_data_attr.shape)
    print('2222', train_label_attr.shape)
    ret_data_attr = []
    ret_label_attr = []
    for i in range(train_data_attr.shape[0]):
        if train_data_attr[i][A] == a:
            ret_data_attr.append(np.concatenate((train_data_attr[i, 0:A], train_data_attr[i, A + 1:])))
            ret_label_attr.append(train_label_attr[i])
    return np.array(ret_data_attr), np.array(ret_label_attr)


def create_tree(*data_set):
    epsilon = 0.1
    train_data_list = data_set[0][0]
    train_label_list = data_set[0][1]
    class_dict = {i for i in train_label_list}
    if len(class_dict) == 1:
        return train_label_list[0]
    if len(class_dict) == 0:
        return major_class(train_label_list)

    ag, epsilon_get = calc_best_feature(train_data_list, train_label_list)
    print('best', ag, epsilon_get)
    if epsilon_get < epsilon:
        return major_class(train_label_list)

    tree_dict = {ag: {}}
    print('tree dict', tree_dict)
    print('train data list', train_data_list.shape)
    print('train label list', train_label_list.shape)
    tree_dict[ag][0] = create_tree(get_sub_data_attr(train_data_list, train_label_list, ag, 0))
    tree_dict[ag][1] = create_tree(get_sub_data_attr(train_data_list, train_label_list, ag, 1))
    return tree_dict


def predict(test_data_list, tree):
    while True:
        (key, value), = tree.items()
        if type(tree[key]).__name__ == 'dict':
            data_val = test_data_list[key]
            del test_data_list[key]
            tree = value[data_val]
            if type(tree).__name__ == 'int':
                return tree
        else:
            return value


def model_test(test_data_list, test_label_list, tree):
    error_cnt = 0
    for i in range(test_data_list.shape[0]):
        if test_label_list[i] != predict(test_data_list[i], tree):
            error_cnt += 1
    return 1 - error_cnt / test_data_list.shape[0]


if __name__ == '__main__':
    # tree = TreeNode()
    train_x, train_y, test_x, test_y = LoadCorpus.load_mnist()
    tree = create_tree((train_x, train_y))
    # print('tree', tree)
    score = model_test(test_x, test_y, tree)
    # print('score', score)
