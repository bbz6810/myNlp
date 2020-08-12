import time

from model.page_rank import PageRank
from tools.load import load_paper_data


class TextRank:
    def __init__(self):
        self.page_rank = PageRank()

    def data_to_tuple(self, dis):
        now = time.time()
        data = load_paper_data()[:100]
        print('加载数据cost:', time.time() - now)
        for index_i, one in enumerate(data):
            for index_y, two in enumerate(one):
                t_dis_len = len(one) if index_y + dis > len(one) else index_y + dis
                for k in range(index_y + 1, t_dis_len):
                    self.page_rank.add_node((two, one[k]))
        print('创建节点cost:', time.time() - now)
        self.page_rank.calc()
        print('训练cost:', time.time() - now)

    def train(self, dis=5):
        self.data_to_tuple(dis)


if __name__ == '__main__':
    textrank = TextRank()
    textrank.train()
