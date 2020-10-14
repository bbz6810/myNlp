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
from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score
import lightgbm as lgb

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

    with open(os.path.join(tianchi_news_class_path, 'tfidf_doc.pkl'), mode='wb') as f:
        pickle.dump(train_term_doc, f)
        pickle.dump(test_term_doc, f)


def f1(y_true, y_pred):
    return f1_score(y_true, y_pred, average='macro')


@running_of_time
def train():
    now = time.time()
    train_df = pd.read_csv(news_classifier_path, sep='\t')
    test_df = pd.read_csv(news_test_path, sep='\t')
    with open(os.path.join(tianchi_news_class_path, 'tfidf_doc.pkl'), mode='rb') as f:
        train_term_doc = pickle.load(f)
        test_term_doc = pickle.load(f)
    print('load corpus done, time', time.time() - now)

    # train_x, test_x, train_y, test_y = train_test_split(train_term_doc, train_df['label'], test_size=0.2, shuffle=True,
    #                                                     random_state=2020)

    # CV 交叉验证
    kf = KFold(n_splits=10, shuffle=True, random_state=2020)
    train_matrix = np.zeros((train_df.shape[0], 14))
    cv_score = []

    # test score is 0.9428105262012133
    # 所有验证 score is 0.9415295480145144
    # function: train, cost time: 59393.73948907852

    # for i, (train_index, valid_index) in enumerate(kf.split(train_term_doc)):
    #     train_xx = train_term_doc[train_index]
    #     train_yy = train_df['label'][train_index]
    #     valid_xx = train_term_doc[valid_index]
    #     valid_yy = train_df['label'][valid_index]
    #
    #     model = lgb.LGBMClassifier(boosting_type='gbdt', num_leaves=2 ** 5, max_depth=-1, n_estimators=2000,
    #                                learning_rate=0.1, objective='multiclass', subsample=0.7, colsample_bytree=0.5,
    #                                reg_lambda=10, num_class=14, random_state=2020, min_child_weight=1.5,
    #                                metric='multi_logloss')
    #     model.fit(train_xx, train_yy, eval_set=(valid_xx, valid_yy), early_stopping_rounds=100)
    #     joblib.dump(model, os.path.join(tianchi_news_class_path, 'tfidf_lgb', 'lgb_k_{}.model'.format(i)))
    #     # 验证集预测
    #     valid_prob = model.predict_proba(valid_xx)
    #     train_matrix[valid_index] = valid_prob.reshape((valid_xx.shape[0], 14))
    #     valid_pred = np.argmax(valid_prob, axis=1)
    #     score = f1(valid_yy, valid_pred)
    #     print('test score is', score)
    #     cv_score.append(score)
    #
    # all_pred = np.argmax(train_matrix, axis=1)
    # score = f1(train_df['label'], all_pred)
    # print('所有验证 score is', score)

    # 测试集预测
    f = 10
    test_pre_matrix = np.zeros((f, test_df.shape[0], 14))
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


if __name__ == '__main__':
    train()
