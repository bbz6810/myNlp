# https://www.biaodianfu.com/lightgbm.html   lgb调参
import os
import time
import sys
import pickle

sys.path.append('/Users/zhoubb/projects/myNlp')

import pandas as pd
import numpy as np
import joblib
import sklearn
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score
import lightgbm as lgb
import xgboost as xgb
from bayes_opt import BayesianOptimization
from heamy.dataset import Dataset
from heamy.estimator import Classifier
from sklearn.svm import SVC

from corpus import news_classifier_path, news_test_path, tianchi_news_class_path
from tools import running_of_time


@running_of_time
def clean_data():
    now = time.time()
    train_df = pd.read_csv(news_classifier_path, sep='\t')
    test_df = pd.read_csv(news_test_path, sep='\t')
    print('load corpus done, time', time.time() - now)

    now = time.time()
    train_df['text_split'] = train_df['text'].apply(lambda x: str(x.split()))
    test_df['text_split'] = test_df['text'].apply(lambda x: str(x.split()))
    print('data apply time', time.time() - now)

    now = time.time()
    word_vec = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), max_df=0.9, min_df=3, use_idf=True,
                               max_features=3000, smooth_idf=True, sublinear_tf=True)
    train_term_doc = word_vec.fit_transform(train_df['text_split'])
    test_term_doc = word_vec.transform(test_df['text_split'])
    print('tf idf fit and transform, time', time.time() - now)

    # print(sorted(word_vec.vocabulary_.items(), key=lambda x: x[1], reverse=True)[-10:])

    with open(os.path.join(tianchi_news_class_path, 'tfidf_doc.pkl'), mode='wb') as f:
        pickle.dump(train_term_doc, f)
        pickle.dump(test_term_doc, f)


def f1(y_true, y_pred):
    return f1_score(y_true, y_pred, average='macro')


@running_of_time
def train_lgb():
    now = time.time()
    train_df = pd.read_csv(news_classifier_path, sep='\t')
    test_df = pd.read_csv(news_test_path, sep='\t')
    with open(os.path.join(tianchi_news_class_path, 'tfidf_doc.pkl'), mode='rb') as f:
        train_term_doc = pickle.load(f)
        test_term_doc = pickle.load(f)
    print('load corpus done, time', time.time() - now)
    print('train doc shape', train_term_doc.shape)

    train_term_doc = train_term_doc[:100]
    test_term_doc = test_term_doc[:100]

    # CV 交叉验证
    # test score is 0.9428105262012133
    # 所有验证 score is 0.9415295480145144
    # function: train, cost time: 59393.73948907852
    kf = KFold(n_splits=10, shuffle=True, random_state=888)
    train_matrix = np.zeros((train_df.shape[0], 14))
    cv_score = []

    for i, (train_index, valid_index) in enumerate(kf.split(train_term_doc)):
        print('k fold', i)
        train_xx, train_yy = train_term_doc[train_index], train_df['label'][train_index]
        valid_xx, valid_yy = train_term_doc[valid_index], train_df['label'][valid_index]

        model = lgb.LGBMClassifier(boosting_type='gbdt', num_leaves=2 ** 5, max_depth=-1, n_estimators=2000,
                                   learning_rate=0.1, objective='multiclass', subsample=0.7, colsample_bytree=0.5,
                                   reg_lambda=10, num_class=14, random_state=2020, min_child_weight=1.5,
                                   metric='multi_logloss')

        model.fit(X=train_xx, y=train_yy, eval_set=(valid_xx, valid_yy), early_stopping_rounds=100, verbose=1)
        joblib.dump(model, os.path.join(tianchi_news_class_path, 'tfidf_lgb', 'lgb_k_{}.model'.format(i)))
        # 验证集预测
        valid_prob = model.predict_proba(valid_xx)
        train_matrix[valid_index] = valid_prob.reshape((valid_xx.shape[0], 14))
        valid_pred = np.argmax(valid_prob, axis=1)
        score = f1(valid_yy, valid_pred)
        print('test score is', score)
        cv_score.append(score)

    all_pred = np.argmax(train_matrix, axis=1)
    score = f1(train_df['label'], all_pred)
    print('所有验证 score is', score)


@running_of_time
def train_xgb():
    now = time.time()
    train_df = pd.read_csv(news_classifier_path, sep='\t')
    test_df = pd.read_csv(news_test_path, sep='\t')
    with open(os.path.join(tianchi_news_class_path, 'tfidf_doc.pkl'), mode='rb') as f:
        train_term_doc = pickle.load(f)
        test_term_doc = pickle.load(f)
    print('load corpus done, time', time.time() - now)
    print('train doc shape', train_term_doc.shape)

    # train_term_doc = train_term_doc[:100]
    # test_term_doc = test_term_doc[:100]

    kf = KFold(n_splits=10, shuffle=True, random_state=888)
    train_matrix = np.zeros((train_df.shape[0], 14))
    cv_score = []

    for i, (train_index, valid_index) in enumerate(kf.split(train_term_doc)):
        print('k fold', i)
        train_xx, train_yy = train_term_doc[train_index], train_df['label'][train_index]
        valid_xx, valid_yy = train_term_doc[valid_index], train_df['label'][valid_index]

        model = xgb.XGBClassifier(learning_rate=0.1,
                                  n_estimators=2000,
                                  booster='gbtree',
                                  objective='multi:softmax',
                                  num_class=14,
                                  max_depth=5,
                                  min_child_weight=1.5,
                                  gamma=0.005,
                                  subsample=0.6,
                                  colsample_bytree=0.5,
                                  reg_alpha=10,
                                  n_jobs=4)

        model.fit(X=train_xx, y=train_yy, verbose=1)
        joblib.dump(model, os.path.join(tianchi_news_class_path, 'tfidf_lgb', 'xgb_k_{}.model'.format(i)))
        # 验证集预测
        valid_prob = model.predict_proba(valid_xx)
        train_matrix[valid_index] = valid_prob.reshape((valid_xx.shape[0], 14))
        valid_pred = np.argmax(valid_prob, axis=1)
        score = f1(valid_yy, valid_pred)
        print('test score is', score)
        cv_score.append(score)

    all_pred = np.argmax(train_matrix, axis=1)
    score = f1(train_df['label'], all_pred)
    print('所有验证 score is', score)


@running_of_time
def train_lr():
    now = time.time()
    train_df = pd.read_csv(news_classifier_path, sep='\t')
    test_df = pd.read_csv(news_test_path, sep='\t')
    with open(os.path.join(tianchi_news_class_path, 'tfidf_doc.pkl'), mode='rb') as f:
        train_term_doc = pickle.load(f)
        test_term_doc = pickle.load(f)
    print('load corpus done, time', time.time() - now)
    # print('train doc shape', train_term_doc.shape)

    # train_term_doc = train_term_doc[:10000]
    # test_term_doc = test_term_doc[:10000]

    kf = KFold(n_splits=10, shuffle=True, random_state=888)
    train_matrix = np.zeros((train_df.shape[0], 14))
    cv_score = []
    # lbfgs  sag
    params = {'C': 8, 'solver': 'sag', 'multi_class': 'multinomial', 'max_iter': 300, 'n_jobs': 1}

    for i, (train_index, valid_index) in enumerate(kf.split(train_term_doc)):
        print('k fold is', i)
        train_xx, train_yy = train_term_doc[train_index], train_df['label'][train_index]
        valid_xx, valid_yy = train_term_doc[valid_index], train_df['label'][valid_index]

        model = LogisticRegression(**params)
        model.fit(X=train_xx, y=train_yy)
        joblib.dump(model, os.path.join(tianchi_news_class_path, 'tfidf_lgb', 'lr_k_{}.model'.format(i)))
        # 验证集预测
        valid_prob = model.predict_proba(valid_xx)
        train_matrix[valid_index] = valid_prob.reshape((valid_xx.shape[0], 14))
        valid_pred = np.argmax(valid_prob, axis=1)
        score = f1(valid_yy, valid_pred)
        print('test f1 is', score, 'params', params)
        cv_score.append(score)

    all_pred = np.argmax(train_matrix, axis=1)
    score = f1(train_df['label'], all_pred)
    print('所有验证 score is', score)


@running_of_time
def train_svm():
    now = time.time()
    train_df = pd.read_csv(news_classifier_path, sep='\t')
    test_df = pd.read_csv(news_test_path, sep='\t')
    with open(os.path.join(tianchi_news_class_path, 'tfidf_doc.pkl'), mode='rb') as f:
        train_term_doc = pickle.load(f)
        test_term_doc = pickle.load(f)
    print('load corpus done, time', time.time() - now)

    kf = KFold(n_splits=10, shuffle=True, random_state=2000)
    train_matrix = np.zeros((train_df.shape[0], 14))
    cv_score = []
    params = {'C': 10, 'kernel': 'rbf', 'gamma': 'scale'}

    for i, (train_index, valid_index) in enumerate(kf.split(train_term_doc)):
        print('k fold is', i)
        train_xx, train_yy = train_term_doc[train_index], train_df['label'][train_index]
        valid_xx, valid_yy = train_term_doc[valid_index], train_df['label'][valid_index]

        model = SVC(**params)
        model.fit(X=train_xx, y=train_yy)
        joblib.dump(model, os.path.join(tianchi_news_class_path, 'tfidf_lgb', 'svc_k_{}.model'.format(i)))
        # 验证集预测
        valid_prob = model.predict_proba(valid_xx)
        train_matrix[valid_index] = valid_prob.reshape((valid_xx.shape[0], 14))
        valid_pred = np.argmax(valid_prob, axis=1)
        score = f1(valid_yy, valid_pred)
        print('test f1 is', score, 'params', params)
        cv_score.append(score)
        return

    all_pred = np.argmax(train_matrix, axis=1)
    score = f1(train_df['label'], all_pred)
    print('所有验证 score is', score)


@running_of_time
def cv_search():
    # xgb 参数详解 https://blog.csdn.net/qq_41076797/article/details/102710299
    now = time.time()
    train_df = pd.read_csv(news_classifier_path, sep='\t')
    # test_df = pd.read_csv(news_test_path, sep='\t')
    with open(os.path.join(tianchi_news_class_path, 'tfidf_doc.pkl'), mode='rb') as f:
        train_term_doc = pickle.load(f)
        # test_term_doc = pickle.load(f)
    print('load corpus done, time', time.time() - now)
    print('train doc shape', train_term_doc.shape)

    param_test = {'max_depth': range(3, 10, 2), 'min_child_weight': range(1, 6, 2)}
    g_search = GridSearchCV(estimator=xgb.XGBClassifier(learning_rate=0.1,
                                                        n_estimators=2000,
                                                        max_depth=-1,
                                                        min_child_weight=1,
                                                        gamma=0,
                                                        subsample=0.8,
                                                        colsample_bytree=0.8),
                            param_grid=param_test, scoring='roc_auc', iid=False, cv=5, verbose=1, n_jobs=4)
    g_search.fit(train_term_doc, train_df['label'].values.reshape(-1, ))
    print(g_search.best_params_)
    print(g_search.best_score_)


@running_of_time
def bayes():
    train_df = pd.read_csv(news_classifier_path, sep='\t')
    with open(os.path.join(tianchi_news_class_path, 'tfidf_doc.pkl'), mode='rb') as f:
        train_term_doc = pickle.load(f)

    def rf_cv_xgb(min_child_weight, gamma, subsample, colsample_bytree, reg_alpha):
        # 建立模型
        model_xgb = xgb.XGBClassifier(
            learning_rate=0.1,
            n_estimators=2000,
            booster='gbtree',
            objective='multi:softmax',
            max_depth=5,
            min_child_weight=1.5,
            gamma=0.005,
            subsample=0.6,
            colsample_bytree=colsample_bytree,
            reg_alpha=reg_alpha,
        )

        val = cross_val_score(model_xgb, train_term_doc, train_df['label'].values.reshape(-1, ), cv=10,
                              scoring='roc_auc', n_jobs=2, verbose=1).mean()
        return val

    bayes_xgb = BayesianOptimization(
        rf_cv_xgb,
        {
            'min_child_weight': (1, 10),
            'gamma': (0, 1),
            'subsample': (0.6, 0.9),
            'colsample_bytree': (0.6, 0.9),
            'reg_alpha': (0, 1),
        }
    )
    bayes_xgb.maximize(n_iter=10)
    # joblib.dump(bayes_lgb, os.path.join(daikuan_path, 'bayes_lgb_model'))
    print(bayes_xgb.max)


def predict():
    # 测试集预测
    now = time.time()
    with open(os.path.join(tianchi_news_class_path, 'tfidf_doc.pkl'), mode='rb') as f:
        train_term_doc = pickle.load(f)
        test_term_doc = pickle.load(f)

    f = 10
    test_pre_matrix = np.zeros((f, test_term_doc.shape[0], 14))
    for model_name in range(f):
        model = joblib.load(os.path.join(tianchi_news_class_path, 'tfidf_lgb', 'lgb_k_{}.model'.format(model_name)))
        test_prob = model.predict_proba(test_term_doc)
        test_pre_matrix[model_name, :, :] = test_prob.reshape((test_term_doc.shape[0], 14))
    test_pred = test_pre_matrix.mean(axis=0)
    test_pred = np.argmax(test_pred, axis=1)
    with open(os.path.join(tianchi_news_class_path, 'sample_submit.csv'), mode='w') as f:
        f.write('label\n')
        for idx, i in enumerate(test_pred):
            f.write('{}\n'.format(i))


@running_of_time
def stack_train():
    train_df = pd.read_csv(news_classifier_path, sep='\t')
    with open(os.path.join(tianchi_news_class_path, 'tfidf_doc.pkl'), mode='rb') as f:
        train_term_doc = pickle.load(f)
        test_term_doc = pickle.load(f)

    k = 10
    preds = []
    for model_name in range(k):
        print('f', model_name)
        model = joblib.load(
            os.path.join(tianchi_news_class_path, 'tfidf_lgb', '0.9438', 'lgb_k_{}.model'.format(model_name)))
        train_prob = model.predict(train_term_doc)
        # train_prob = model.predict(test_term_doc)
        preds.append(train_prob.reshape(-1, 1))

        model = joblib.load(os.path.join(tianchi_news_class_path, 'tfidf_lgb', 'xgb_k_{}.model'.format(model_name)))
        train_prob = model.predict(train_term_doc)
        # train_prob = model.predict(test_term_doc)
        preds.append(train_prob.reshape(-1, 1))

    # xgb_pre_matrix = np.zeros((f, test_df.shape[0], 14))
    # for model_name in range(f):
    #     model = joblib.load(os.path.join(tianchi_news_class_path, 'tfidf_lgb', 'xgb_k_{}.model'.format(model_name)))
    #     test_prob = model.predict_proba(test_term_doc)
    #     xgb_pre_matrix[model_name, :, :] = test_prob.reshape((test_term_doc.shape[0], 14))

    merge = np.concatenate(preds, axis=-1)
    print('合并后的shape', merge.shape)
    with open(os.path.join(tianchi_news_class_path, 'tfidf_lgb', 'lgb_xgb_train_kfold_10.npy'), mode='wb') as f:
        pickle.dump(merge, f)


def predict_train():
    train_df = pd.read_csv(news_classifier_path, sep='\t')
    with open(os.path.join(tianchi_news_class_path, 'tfidf_lgb', 'lgb_xgb_train_kfold_10.npy'), mode='rb') as f:
        merge = pickle.load(f)

    c = 0
    train_pred = []
    for one in merge:
        d = {}
        for i in one:
            if i in d:
                d[i] += 1
            else:
                d[i] = 1
        d = sorted(d.items(), key=lambda x: x[1], reverse=True)
        if len(d) >= 2 and d[0][1] == d[1][1]:
            print(d)
            c += 1
        train_pred.append(d[0][0])
    score = f1_score(train_df['label'].values, np.array(train_pred), average='macro')
    print('merge train f1 score', score)

    # x, y = merge, train_df['label'].values
    # # train_x, train_y, test_x, test_y = x[:150000], y[:150000], x[150000:], y[150000:]
    # train_x, test_x, train_y, test_y = train_test_split(x, y, train_size=0.8, shuffle=True)

    # 0.9492
    # lr = LogisticRegression(C=10, solver='sag', multi_class='multinomial')
    # lr.fit(train_x, train_y)
    # joblib.dump(lr, os.path.join(tianchi_news_class_path, 'tfidf_lgb', 'lr.model'))
    # score = lr.score(test_x, test_y)
    # print('score', score)

    # # 0.98502
    # svc = SVC(C=10, kernel='rbf')
    # svc.fit(train_x, train_y)
    # joblib.dump(svc, os.path.join(tianchi_news_class_path, 'tfidf_lgb', 'svc.model'))
    # score = svc.score(test_x, test_y)
    # print('score', score)


@running_of_time
def stack_test():
    train_df = pd.read_csv(news_classifier_path, sep='\t')
    with open(os.path.join(tianchi_news_class_path, 'tfidf_doc.pkl'), mode='rb') as f:
        train_term_doc = pickle.load(f)
        test_term_doc = pickle.load(f)

    k = 10
    preds = []
    for model_name in range(k):
        print('f', model_name)
        model = joblib.load(
            os.path.join(tianchi_news_class_path, 'tfidf_lgb', '0.9438', 'lgb_k_{}.model'.format(model_name)))
        train_prob = model.predict(test_term_doc)
        # train_prob = model.predict(test_term_doc)
        preds.append(train_prob.reshape(-1, 1))

        model = joblib.load(os.path.join(tianchi_news_class_path, 'tfidf_lgb', 'xgb_k_{}.model'.format(model_name)))
        train_prob = model.predict(test_term_doc)
        # train_prob = model.predict(test_term_doc)
        preds.append(train_prob.reshape(-1, 1))

    merge = np.concatenate(preds, axis=-1)
    print('合并后的shape', merge.shape)
    with open(os.path.join(tianchi_news_class_path, 'tfidf_lgb', 'lgb_xgb_test_kfold_10.npy'), mode='wb') as f:
        pickle.dump(merge, f)


def predict_test():
    with open(os.path.join(tianchi_news_class_path, 'tfidf_lgb', 'lgb_xgb_test_kfold_10.npy'), mode='rb') as f:
        merge = pickle.load(f)

    c = 0
    test_pred = []
    for one in merge:
        d = {}
        for i in one:
            if i in d:
                d[i] += 1
            else:
                d[i] = 1
        d = sorted(d.items(), key=lambda x: x[1], reverse=True)
        if len(d) >= 2 and d[0][1] == d[1][1]:
            print(d)
            c += 1
        test_pred.append(d[0][0])
    print(c)

    with open(os.path.join(tianchi_news_class_path, 'sample_submit.csv'), mode='w') as f:
        f.write('label\n')
        for idx, i in enumerate(test_pred):
            f.write('{}\n'.format(int(round(i))))


def test_result():
    fa = os.path.join(tianchi_news_class_path, 'sample_submit.csv')
    fb = os.path.join(tianchi_news_class_path, 'sample_submit_0.9447.csv')

    a = open(fa, mode='r')
    b = open(fb, mode='r')

    right = 0
    for i, j in zip(a.readlines(), b.readlines()):
        i, j = i.strip(), j.strip()
        if i == j:
            right += 1
    print(right / 50000)


if __name__ == '__main__':
    # clean_data()
    # train()
    # train_xgb()
    # train_lr()
    # train_svm()
    # bayes()
    # stack()
    # predict_train()
    # stack_test()
    predict_test()
    test_result()
