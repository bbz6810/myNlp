# https://www.biaodianfu.com/lightgbm.html   lgb调参
import os
import time
import pickle

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

    # CV
    kf = KFold(n_splits=10, shuffle=True, random_state=2020)
    train_matrix = np.zeros((train_df.shape[0], 14))
    test_pre_matrix = np.zeros((10, test_df.shape[0], 14))
    cv_score = []

    for i, (train_index, test_index) in enumerate(kf.split(train_term_doc)):
        train_xx = train_term_doc[train_index]
        train_yy = train_df['label'][train_index]
        test_xx = train_term_doc[test_index]
        test_yy = train_df['label'][test_index]

        model = lgb.LGBMClassifier(boosting_type='gbdt', num_leaves=2 ** 5, max_depth=-1, n_estimators=2000,
                                   learning_rate=0.1, objective='multiclass', subsample=0.7, colsample_bytree=0.5,
                                   reg_lambda=10, num_class=14, random_state=2020, min_child_weight=1.5,
                                   metric='multi_logloss')
        model.fit(train_xx, train_yy, eval_set=(test_xx, test_yy), early_stopping_rounds=100)
        joblib.dump(model, os.path.join(tianchi_news_class_path, 'lgb_k_{}.model'.format(i)))
        # 验证集预测
        test_prob = model.predict_proba(test_xx)
        train_matrix[test_index] = test_prob.reshape((test_xx.shape[0], 14))
        test_pred = np.argmax(test_prob, axis=1)
        score = f1(test_yy, test_pred)
        print('test score is', score)
        cv_score.append(score)

    # 测试集预测
    # test2_prob = model.predict_proba(test_term_doc)
    # test_pre_matrix[i, :, :] = test2_prob.reshape((test_term_doc.shape[0], 14))

    all_pred = np.argmax(train_matrix, axis=1)
    score = f1(train_df['label'], all_pred)
    print('所有验证 score is', score)


if __name__ == '__main__':
    # clean_data()
    train()
