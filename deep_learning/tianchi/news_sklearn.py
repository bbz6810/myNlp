import os
import time
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import joblib
from corpus import news_classifier_path, news_test_path, news_one_to_one_path, corpus_root_path
from tools import running_of_time

"""
    1、特征值选择，pca主成分分析
    2、根据主成分重新生成特征数据
    3、选择分类模型svm，rf，lr
    4、调优
    
    solver：优化算法选择参数
    liblinear：使用了开源的liblinear库实现，内部使用了坐标轴下降法来迭代优化损失函数。
    lbfgs：拟牛顿法的一种，利用损失函数二阶导数矩阵即海森矩阵来迭代优化损失函数。
    newton-cg：也是牛顿法家族的一种，利用损失函数二阶导数矩阵即海森矩阵来迭代优化损失函数。只用于L2
    sag：即随机平均梯度下降，是梯度下降法的变种，和普通梯度下降法的区别是每次迭代仅仅用一部分的样本来计算梯度，适合于样本数据多的时候。只用于L2
    saga：线性收敛的随机优化算法的的变重。只用于L2
    
    1、把训练集按标签分类，生成一对一的结果
    
"""


@running_of_time
def format_test_data(data_path):
    x = []
    with open(data_path, mode='r') as f:
        for line in f.readlines()[1:]:
            x.append(line.strip())
    return x


@running_of_time
def format_data_tf_idf_one2one(data_path):
    x, y = [], []
    with open(data_path, mode='r') as f:
        for line in f.readlines()[1:]:
            line = line.strip().split('\t')
            if line[0] in ('0', '1', '2', '3'):
                x.append(line[1])
                y.append(int(line[0]))
    return x, y


class News:
    def __init__(self):
        pass

    @running_of_time
    def load_train_data(self, data_path):
        data_dict = dict()
        with open(data_path, mode='r') as f:
            for line in f.readlines()[1:]:
                line = line.strip().split('\t')
                if line[0] not in data_dict:
                    data_dict[line[0]] = []
                data_dict[line[0]].append([line[1], int(line[0])])
        return data_dict

    def load_test_data(self, data_path):
        x = []
        with open(data_path, mode='r') as f:
            for line in f.readlines()[1:]:
                x.append(line.strip())
        return x

    def train(self):
        data_dict = self.load_train_data(news_classifier_path)
        for k1 in data_dict:
            for k2 in data_dict:
                if k1 == k2:
                    continue
                self.train_one2one(data_dict, k1, k2)

    def train_one2one(self, data_dict, k1, k2):
        x = data_dict[k1][0] + data_dict[k2][0]
        y = data_dict[k1][1] + data_dict[k2][1]
        train_x, test_x, train_y, test_y = train_test_split(x, y)


@running_of_time
def sklearn_lr(train_x, train_y, test_x, test_y, test, solver, c):
    lr = LogisticRegression(penalty='l2', solver=solver, C=c, multi_class='auto')
    lr.fit(train_x, train_y)
    joblib.dump(lr, os.path.join(news_one_to_one_path, 'lr_model'))
    score = lr.score(test_x, test_y)
    print('lr score', score)
    s = lr.predict(test)
    for i in s:
        print(i)


def sklearn_svc(train_x, train_y, test_x, test_y, ker, c):
    svc = SVC(C=c, kernel=ker)
    svc.fit(train_x, train_y)
    joblib.dump(svc, os.path.join(news_one_to_one_path, 'svc_model'))
    score = svc.score(test_x, test_y)
    print('svc score', score)


def sklearn_rf(train_x, train_y, test_x, test_y):
    rf = RandomForestClassifier(n_estimators=100, max_depth=4)
    rf.fit(train_x, train_y)
    joblib.dump(rf, os.path.join(news_one_to_one_path, 'rf_model'))
    score = rf.score(test_x, test_y)
    print('rf score', score)


def load_lr():
    lr = joblib.load(os.path.join(news_one_to_one_path, 'lr_model'))
    return lr


def load_svc():
    svc = joblib.load(os.path.join(news_one_to_one_path, 'svc_model'))
    return svc


def load_rf():
    rf = joblib.load(os.path.join(news_one_to_one_path, 'rf_model'))
    return rf


@running_of_time
def train_one(retv, y, train_len, component, solver, c):
    print('source shape', retv.toarray().shape)
    now = time.time()
    pca = PCA(n_components=component)
    tmp_x = pca.fit_transform(retv.toarray())
    train = tmp_x[:train_len]
    test = tmp_x[train_len:]
    np.save(os.path.join(corpus_root_path, 'news_test'))
    print('pca shape', tmp_x.shape, '\npca fix time', time.time() - now)
    train_x, test_x, train_y, test_y = train_test_split(train, y, train_size=0.8)
    sklearn_lr(train_x, train_y, test_x, test_y, test, solver, c)


def train():
    x, y = format_data_tf_idf_one2one(news_classifier_path)
    x2 = format_test_data(news_test_path)
    print('train lines', len(x))
    print('test lines', len(x2))
    tfidf = TfidfVectorizer()
    now = time.time()
    retv = tfidf.fit_transform(x + x2)
    print('tfidf fix time', time.time() - now)
    train_one(retv, y, len(x), 1024, 'sag', 2)

    # pca_n_com = (1024,)
    # solver = ('liblinear', 'sag')
    # kernel = ('linear', 'rbf')
    # cs = (8,)
    # for com in pca_n_com:
    #     for sol in solver:
    #         for c in cs:
    #             print('=======================', com, sol, c)
    #             train_one(retv, y, len(x), com, sol, c)


@running_of_time
def train_save():
    x, y = format_data_tf_idf_one2one(news_classifier_path)
    # x2 = format_test_data(news_test_path)
    print('train x len', len(x))
    now = time.time()
    tfidf = TfidfVectorizer()
    tfidf.fit(x)
    print('tfidf fit time', time.time() - now)
    joblib.dump(tfidf, os.path.join(news_one_to_one_path, 'tfidf_model'))
    now = time.time()
    retv = tfidf.transform(x)
    print('tfidf transform time', time.time() - now)
    now = time.time()
    pca = PCA(n_components=1024)
    pca.fit(retv.toarray())
    print('pca fit time', time.time() - now)
    joblib.dump(pca, os.path.join(news_one_to_one_path, 'pca_model'))

    tfidf = joblib.load(os.path.join(news_one_to_one_path, 'tfidf_model'))
    pca = joblib.load(os.path.join(news_one_to_one_path, 'pca_model'))
    now = time.time()
    retv = tfidf.transform(x)
    print('tfidf transform time', time.time() - now)
    now = time.time()
    tmp_x = pca.transform(retv.toarray())
    print('pca transform time', time.time() - now)

    train_x, test_x, train_y, test_y = train_test_split(tmp_x, y, train_size=0.6)

    lr = LogisticRegression(penalty='l2', solver='sag', C=10, max_iter=100)
    # 3 class lr score 0.9718102734051367
    # 2 class lr score 0.9659914758996441
    # 2 class lr score 0.9754168588940881,sag,c=10
    # 3 class lr score 0.9780501444682635,sag,c=100
    # 4 class lr score 0.9688230408159323,sag,c=100,n=512
    # 4 class lr score 0.9724545577469141,sag,c=100,n=1024
    now = time.time()
    lr.fit(train_x, train_y)
    print('lr fit time', time.time() - now)
    joblib.dump(lr, os.path.join(news_one_to_one_path, 'lr_model'))

    lr = load_lr()
    score = lr.score(test_x, test_y)
    print('lr score', score)


if __name__ == '__main__':
    # train()
    # train_save()
    from deep_learning.tianchi.daikuan_classifier import DaiKuan
    from corpus import daikuan_classifier_path, daikuan_test_path

    dai = DaiKuan()
    dai.format_train_x_train_y_test_x(daikuan_classifier_path, daikuan_test_path)
