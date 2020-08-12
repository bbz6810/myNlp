import time
import sys
import numpy as np


class PageRank:
    def __init__(self):
        self.d = 0.85
        self.w = None
        self.mat = None
        self.pr_dict = dict()
        self.key_index = dict()

    def add_nodes(self, nodes):
        for node in nodes:
            self.add_node(node)

    def add_node(self, node):
        if node[0] in self.pr_dict:
            if node[1] in self.pr_dict[node[0]]:
                self.pr_dict[node[0]][node[1]] += 1
            else:
                self.pr_dict[node[0]][node[1]] = 1
        else:
            self.pr_dict[node[0]] = dict()
            self.pr_dict[node[0]][node[1]] = 1

    def create_key_index(self):
        for key, value in self.pr_dict.items():
            if key not in self.key_index:
                self.key_index[key] = len(self.key_index)
            for key2 in value.keys():
                if key2 not in self.key_index:
                    self.key_index[key2] = len(self.key_index)

    def create_mat_w(self):
        self.create_key_index()
        self.mat = np.zeros(shape=(len(self.key_index), len(self.key_index)), dtype='float16')
        print('mat shape', self.mat.shape)
        self.w = np.array([[1 / len(self.key_index)] for _ in range(len(self.key_index))], dtype='float16')
        print('w size:(k)', sys.getsizeof(self.w) / 1024)
        print('mat size:(M)', sys.getsizeof(self.mat) / 1024 / 1024)
        for key, value in self.pr_dict.items():
            sum_value2 = sum(value.values())
            for key2, value2 in value.items():
                self.mat[self.key_index[key]][self.key_index[key2]] = value2 / sum_value2

    def calc(self, epoch=100):
        self.create_mat_w()
        old_w = self.w[:]
        for _ in range(epoch):
            now = time.time()
            t_w = self.w[:]
            self.w = self.d * np.dot(self.mat.T, self.w) + (1 - self.d) * old_w
            if (t_w == self.w).all():
                print('end epoch', _)
                break
            print('train w', _, time.time() - now)
        index_key = {value: key for key, value in self.key_index.items()}
        w_dict = sorted({index_key[i]: value[0] for i, value in enumerate(self.w)}.items(), key=lambda x: x[1],
                        reverse=True)
        print(w_dict[:50])


if __name__ == '__main__':
    pagerank = PageRank()
    edges = [("A", "B"), ("A", "C"), ("A", "D"), ("B", "A"), ("B", "D"), ("C", "A"), ("D", "B"), ("D", "C")]
    pagerank.add_nodes(edges)
    pagerank.calc()
