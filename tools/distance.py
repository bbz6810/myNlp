import numpy as np
import math


def euclidean_distance(x1, x2):
    """欧式距离

    :param x1:
    :param x2:
    :return:
    """
    if isinstance(x1, np.ndarray):
        return np.sqrt(np.dot((x1 - x2), (x1 - x2).T))
    else:
        return math.sqrt(sum(math.pow((i - j), 2) for i, j in zip(x1, x2)))
