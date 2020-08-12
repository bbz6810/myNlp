from corpus import paper_path
from tools import delete_punctuation, filter_stop


def load_paper_data():
    data_list = []
    with open(paper_path, encoding='gb2312', mode='r', errors='ignore') as f:
        for line in f.readlines():
            t = filter_stop(filter(lambda x: x != '', map(lambda x: delete_punctuation(x.split('/')[0]), line.split()[1:])))
            if t:
                data_list.append(t)
    return data_list
