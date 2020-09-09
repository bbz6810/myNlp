import numpy as np
import pandas as pd

from tools import running_of_time
from corpus import daikuan_classifier_path, daikuan_test_path


@running_of_time
def format_data(data_path):
    title = []
    x, y = [], []
    data = pd.read_csv(data_path)
    # with open(data_path, mode='r') as f:
    #     for idx, line in enumerate(f.readlines()[:10]):
    #         # print(line)
    #         if idx == 0:
    #             title = line.strip().split(',')
    #         else:
    #             x.append(line.strip().split(','))

    s = set([len(i) for i in x])
    print(s)
    s = set([i[title.index('delinquency_2years')] for i in x])
    print(s)
    return x, y


if __name__ == '__main__':
    format_data(daikuan_classifier_path)
