import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
from corpus import news_classifier_path, news_test_path, news_one_to_one_path
from tools import running_of_time

"""
    1、特征值选择，pca主成分分析
    2、根据主成分重新生成特征数据
    3、选择分类模型svm，rf，lr
    4、调优
"""


@running_of_time
def format_data(data_path):
    x, y = [], []
    with open(data_path, mode='r') as f:
        for line in f.readlines()[1:1000]:
            line = line.strip().split('\t')
            # print(line)
            x.append(list(map(int, line[1].split())))
            y.append(int(line[0]))
    return x, y


@running_of_time
def format_test_data(data_path):
    x = []
    with open(data_path, mode='r') as f:
        for line in f.readlines()[1:]:
            line = line.strip().split('\t')
            x.append(line[1])
    return x


@running_of_time
def format_data_tf_idf_one2one(data_path):
    x, y = [], []
    with open(data_path, mode='r') as f:
        for line in f.readlines()[1:]:
            line = line.strip().split('\t')
            if line[0] in ('0', '1'):
                x.append(line[1])
                y.append(int(line[0]))
    return x, y


def sklearn_lr(train_x, train_y, test_x, test_y, test, solver, c):
    lr = LogisticRegression(penalty='l2', solver=solver, C=c)
    lr.fit(train_x, train_y)
    joblib.dump(lr, os.path.join(news_one_to_one_path, 'lr_model'))
    score = lr.score(test_x, test_y)
    print('lr score', score)
    # s = lr.predict(test)
    # print(s)


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
    # print('source shape', retv.toarray().shape)
    pca = PCA(n_components=component)
    tmp_x = pca.fit_transform(retv.toarray())
    train = tmp_x[:train_len]
    test = tmp_x[train_len:]
    # print('pca shape', tmp_x.shape)
    train_x, test_x, train_y, test_y = train_test_split(train, y, train_size=0.8)
    sklearn_lr(train_x, train_y, test_x, test_y, test, solver, c)


def run():
    x, y = format_data_tf_idf_one2one(news_classifier_path)
    x2 = format_test_data(news_test_path)
    print('train lines', len(x))
    print('test lines', len(x2))
    tfidf = TfidfVectorizer()
    retv = tfidf.fit_transform(x + x2)

    pca_n_com = (128, 256, 512, 1024, 2048)
    solver = ('newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga')
    kernel = ('linear', 'rbf')
    cs = (1, 10, 100)
    for com in pca_n_com:
        for sol in solver:
            for c in cs:
                print('=======================', com, sol, c)
                train_one(retv, y, len(x), com, sol, c)


if __name__ == '__main__':
    run()
