import time
import numpy as np

from corpus.load_corpus import LoadCorpus


def calc_e_gx(train_data_attr, train_label_attr, n, div, rule, d):
    e = 0
    x = train_data_attr[:, n]
    y = train_label_attr
    predict = []

    if rule == 'LisOne':
        l = 1
        h = -1
    else:
        l = -1
        h = 1

    for i in range(train_data_attr.shape[0]):
        if x[i] < div:
            predict.append(l)
            if y[i] != l:
                e += d[i]
        else:
            predict.append(h)
            if y[i] != h:
                e += d[i]
    return np.array(predict), e


def create_sigle_boosting_tree(train_data_attr, train_label_attr, d):
    m, n = np.shape(train_data_attr)
    sigle_boost_tree = dict()
    sigle_boost_tree['e'] = 1
    for i in range(n):
        for div in [-0.5, 0.5, 1.5]:
            for rule in ['LisOne', 'HisOne']:
                gx, e = calc_e_gx(train_data_attr, train_label_attr, i, div, rule, d)
                if e < sigle_boost_tree['e']:
                    sigle_boost_tree['e'] = e
                    sigle_boost_tree['div'] = div
                    sigle_boost_tree['rule'] = rule
                    sigle_boost_tree['gx'] = gx
                    sigle_boost_tree['feature'] = 1
    return sigle_boost_tree


def create_boost_tree(train_data_list, train_label_list, tree_num=50):
    train_data_attr = np.array(train_data_list)
    train_label_attr = np.array(train_label_list)
    finall_predict = np.zeros(train_label_attr.shape[0])
    m, n = np.shape(train_data_attr)

    d = [1 / m] * m
    tree = []

    for i in range(tree_num):
        cur_tree = create_sigle_boosting_tree(train_data_attr, train_label_attr, d)
        alpha = 1 / 2 * np.log((1 - cur_tree['e']) / cur_tree['e'])
        gx = cur_tree['gx']
        d = np.multiply(d, np.exp(-1 * alpha * np.multiply(train_label_attr, gx))) / sum(d)
        cur_tree['alpha'] = alpha
        tree.append(cur_tree)

        # finall_predict += alpha * gx
        # error = sum([1 for j in range(train_data_attr.shape[0]) if np.sign(finall_predict[i]) != train_label_attr[i]])
        # finall_error = error / train_data_attr.shape[0]
        # if finall_error == 0:
        #     return tree

        score = model_test(test_x, test_y, tree)
        print('iter', i, cur_tree, score)
    return tree


def predict(x, div, rule, feature):
    if rule == 'LisOne':
        l = 1
        h = -1
    else:
        l = -1
        h = 1
    if x[feature] < div:
        return l
    else:
        return h


def model_test(test_data_list, test_label_list, tree):
    error_cnt = 0
    for i in range(test_data_list.shape[0]):
        result = 0
        for cur_tree in tree:
            div, rule, feature, alpha = cur_tree['div'], cur_tree['rule'], cur_tree['feature'], cur_tree['alpha']
            result += alpha * predict(test_data_list[i], div, rule, feature)
        if np.sign(result) != test_label_list[i]:
            error_cnt += 1
    return 1 - error_cnt / test_data_list.shape[0]


if __name__ == '__main__':
    train_x, train_y, test_x, test_y = LoadCorpus.load_mnist()
    location = np.random.permutation(len(train_x))
    train_x = train_x[location]
    train_y = train_y[location]
    tree = create_boost_tree(train_x, train_y)
    score = model_test(test_x, test_y, tree)
    print('score', score)
