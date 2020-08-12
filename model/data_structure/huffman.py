"""Huffman编码
    1、带权树
    2、构造Huffman编码
"""

from copy import deepcopy
from corpus import wechat_new_word_data_path


def read_data():
    d = dict()
    with open(wechat_new_word_data_path, encoding='utf8', mode='r') as f:
        for line in f.readlines():
            t = line.strip().split(',')
            d[t[0]] = int(t[1])
    return d


class HuffmanNode:
    def __init__(self, key=None, value=None):
        self.key = key
        self.value = value
        self.left = None
        self.right = None

    def __str__(self):
        return 'key:{}, value:{}, left:{}, right:{}'.format(self.key, self.value, self.left, self.right)


class Huffman:
    def __init__(self):
        self.data_dict = read_data()

    def create_huffman_node(self, key, value):
        return HuffmanNode(key=key, value=value)

    def create_huffman_node_list(self, data_dict):
        return [HuffmanNode(key, value) for key, value in data_dict.items()]

    def build_tree(self, node_list):
        lens = 0
        for i in node_list:
            lens += len(i.key)
        while len(node_list) > 1:
            node_list = sorted(node_list, key=lambda x: x.value, reverse=False)
            left, right = deepcopy(node_list[0]), deepcopy(node_list[1])
            new_node = self.create_huffman_node(None, value=(left.value + right.value))
            new_node.left, new_node.right = left, right
            node_list = node_list[2:]
            node_list.append(new_node)
        huffman_tree_code = dict()

        def loop(p, v):
            if not p:
                return
            if p.key:
                huffman_tree_code[p.key] = ''.join(map(lambda x: str(x), v[1:]))
            loop(p.left, v + [0])
            loop(p.right, v + [1])

        loop(node_list[0], [-1])

        print(huffman_tree_code)


if __name__ == '__main__':
    huffman = Huffman()
    node_list = huffman.create_huffman_node_list(read_data())
    huffman.build_tree(node_list[:10])
