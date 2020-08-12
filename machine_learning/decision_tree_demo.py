from sklearn.datasets import load_iris

from model.decision_tree import create_tree
from tools import print_dict

feature_label = {0: '形状', 1: '敲声', 2: '纹理', 3: '根茎'}
label_feature = {v: k for k, v in feature_label.items()}


def test():
    data = load_iris()
    data_list = [[x, y] for x, y in zip(data.data, data.target)]
    print('数据总共', len(data_list))
    tree = create_tree(data_list, [], feature_label)
    print(label_feature)
    # print('tree', tree)
    # print_dict(tree, 0)
    # print(tree.keys())
    # print(tree.values())
    predict(data_list, tree)


def predict(data_list, tree):
    for data in data_list:
        row, c = data
        c2 = predict_one(row, tree)
        print('真实值{}，预测值{}'.format(c, c2))


def predict_one(data_one, tree):
    node = tree
    attr = None
    for key in node.keys():
        if key in label_feature:
            attr = key
            break
    attr_index = label_feature[attr]
    while True:
        if data_one[attr_index] in node[attr]:
            t = node[attr][data_one[attr_index]]
            if isinstance(t, dict):
                node = t
                for key in node.keys():
                    if key in label_feature:
                        attr = key
                        break
                attr_index = label_feature[attr]
            else:
                return t
        else:
            print('未找到该值')
            return


if __name__ == '__main__':
    test()
