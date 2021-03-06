import os
import sys
import time
import datetime
import joblib
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier, VotingClassifier, RandomForestClassifier
from lightgbm import LGBMClassifier
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn import metrics
from sklearn.feature_selection import VarianceThreshold
from tqdm import tqdm
from bayes_opt import BayesianOptimization
from sklearn.metrics import f1_score
from keras import layers, models
from keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel

sys.path.append('/Users/zhoubb/projects/myNlp')

from tools import running_of_time
from corpus import daikuan_classifier_path, daikuan_test_path, corpus_root_path, daikuan_path

"""
    1、删除错误列数据并返回x和y
    2、处理离散化数据
    3、处理连续数据
    4、拆分训练和测试数据，并入模型跑数据
    
    贝叶斯调参能调到一个较优的解，要根据网格搜索具体细调
"""

title = ['id', 'loanAmnt', 'term', 'interestRate', 'installment', 'grade', 'subGrade', 'employmentTitle',
         'employmentLength', 'homeOwnership', 'annualIncome', 'verificationStatus', 'issueDate', 'isDefault',
         'purpose', 'postCode', 'regionCode', 'dti', 'delinquency_2years', 'ficoRangeLow', 'ficoRangeHigh',
         'openAcc', 'pubRec', 'pubRecBankruptcies', 'revolBal', 'revolUtil', 'totalAcc', 'initialListStatus',
         'applicationType', 'earliesCreditLine', 'title', 'policyCode', 'n0', 'n1', 'n2', 'n2', 'n4', 'n5', 'n6',
         'n7', 'n8', 'n9', 'n10', 'n11', 'n12', 'n13', 'n14']

normal = ['loanAmnt', 'term', 'interestRate', 'installment', 'employmentTitle',
          'homeOwnership', 'annualIncome', 'verificationStatus',
          'purpose', 'postCode', 'regionCode', 'dti', 'delinquency_2years', 'ficoRangeLow', 'ficoRangeHigh',
          'openAcc', 'pubRec', 'pubRecBankruptcies', 'revolBal', 'revolUtil', 'totalAcc', 'initialListStatus',
          'applicationType', 'title', 'policyCode', 'n0', 'n1', 'n2', 'n2', 'n4', 'n5', 'n6',
          'n7', 'n8', 'n9', 'n10', 'n11', 'n12', 'n13', 'n14']

employment_length_dict = {'< 1 year': 0, '10+ years': 10, '8 years': 8, '1 year': 1, '2 years': 2, '5 years': 5,
                          '4 years': 4, '3 years': 3, '6 years': 6, '7 years': 7, '9 years': 9}

train_max_length = 800000
test_max_length = 800000
samples = 100000
source_x_dim = 144


def find_outliers_by_3segama(data, fea):
    data_std = np.std(data[fea])
    data_mean = np.mean(data[fea])
    outliers_cut_off = data_std * 3
    lower = data_mean - outliers_cut_off
    upper = data_mean + outliers_cut_off
    data[fea + '_outliers'] = data[fea].apply(lambda x: '异常值' if x > upper or x < lower else '正常值')
    return data


class DaiKuan:
    def __init__(self):
        pass

    @running_of_time
    def format_train_x_train_y_test_x2(self, train_data_path, test_data_path):
        # train_data = pd.read_csv(train_data_path)[:train_max_length]
        # test_data = pd.read_csv(test_data_path)[:test_max_length]

        # print('数据简介\n', train_data.describe())
        # print(test_data.describe())

        # print('查看为空的情况')
        # print(train_data.isnull().sum())
        # print(test_data.isnull().sum())

        # train_data.info()
        # columns = train_data.columns.values
        # for column in columns:
        #     # print(train_data[column].value_counts())
        #     print(train_data[column].nunique())
        train_x, train_y, test = self.load_train_x_train_y_test_x(balance=False)

        # lr = LogisticRegression(solver='sag', C=10)
        # lr.fit(train_x, train_y)
        # print(lr.coef_)

        # clf = ExtraTreesClassifier(n_estimators=500, max_depth=19)
        # clf = clf.fit(train_x, train_y)
        # clf.feature_importances_
        # model = SelectFromModel(clf, prefit=True)
        # train_x_new = model.transform(train_x)
        # test_x_new = model.transform(test)
        # print(train_x_new.shape)
        # print(test_x_new.shape)
        #
        # np.save(os.path.join(daikuan_path, 'train_data_stand_s'), train_x_new)
        # np.save(os.path.join(daikuan_path, 'test_data_stand_s'), test_x_new)
        return

    @running_of_time
    def format_train_x_train_y_test_x(self, train_data_path, test_data_path):
        train_data = pd.read_csv(train_data_path)[:train_max_length]
        test_data = pd.read_csv(test_data_path)[:test_max_length]

        # 删掉多余的行
        train_data.pop('n2.1')
        test_data.pop('n2.1')
        test_data.pop('n2.2')
        test_data.pop('n2.3')

        # all_data = pd.concat([train_data, test_data], ignore_index=True)

        ## 数据分析
        # print(train_data.columns)
        # print(train_data.isnull().any().sum())
        # print(test_data.isnull().any().sum())
        # 缺失值占比
        # missing = train_data.isnull().sum() / len(train_data)
        # missing = missing[missing > 0]
        # missing.sort_values(inplace=True)
        # print(missing)
        # 特征属性为1的值
        # one_value_fea = [col for col in train_data.columns if train_data[col].nunique() <= 1]
        # one_value_fea_test = [col for col in test_data.columns if test_data[col].nunique() <= 1]
        # print(one_value_fea, one_value_fea_test)
        # 类别型和数值型属性
        numerical_fea = list(train_data.select_dtypes(exclude=['object']).columns)
        category_fea = list(filter(lambda x: x not in numerical_fea, list(train_data.columns)))
        print('数值型', numerical_fea)
        print('类别型', category_fea)
        numerical_fea.remove('isDefault')
        # print(numerical_fea)
        # # print(category_fea)
        # # 连续和离散型变量
        # numerical_serial_fea = []  # 连续
        # numerical_noserial_fea = []  # 离散
        # for fea in numerical_fea:
        #     temp = train_data[fea].nunique()
        #     if temp <= 10:
        #         numerical_noserial_fea.append(fea)
        #     else:
        #         numerical_serial_fea.append(fea)
        # print(numerical_serial_fea)
        # print(numerical_noserial_fea)
        # 数值类别变量分析
        # print(train_data['term'].value_counts())
        # print(train_data['homeOwnership'].value_counts())
        # print(train_data['verificationStatus'].value_counts())
        # print(train_data['initialListStatus'].value_counts())
        # print(train_data['applicationType'].value_counts())
        # print(train_data['policyCode'].value_counts())
        # print(train_data['n11'].value_counts())
        # print(train_data['n12'].value_counts())
        # 非数值变量分析
        # for fea in category_fea:
        #     print(train_data[fea].value_counts())

        ## 特征选择
        # 空值填充:平均数
        train_data[numerical_fea] = train_data[numerical_fea].fillna(train_data[numerical_fea].median())
        test_data[numerical_fea] = test_data[numerical_fea].fillna(test_data[numerical_fea].median())
        # 离散类型用众数
        train_data[category_fea] = train_data[category_fea].fillna(train_data[category_fea].mode())
        test_data[category_fea] = test_data[category_fea].fillna(test_data[category_fea].mode())

        # 时间格式处理
        train_data['issueDate'] = pd.to_datetime(train_data['issueDate'], format='%Y-%m-%d')
        test_data['issueDate'] = pd.to_datetime(test_data['issueDate'], format='%Y-%m-%d')
        startdate = datetime.datetime.strptime('2007-06-01', '%Y-%m-%d')
        train_data['issueDate'] = train_data['issueDate'].apply(lambda x: x - startdate).dt.days
        test_data['issueDate'] = test_data['issueDate'].apply(lambda x: x - startdate).dt.days
        # print(train_data['employmentLength'].value_counts(dropna=False).sort_index())
        train_data['employmentLength'] = train_data['employmentLength'].map(employment_length_dict)
        test_data['employmentLength'] = test_data['employmentLength'].map(employment_length_dict)
        # print(train_data['employmentLength'].value_counts(dropna=False).sort_index())
        train_data['earliesCreditLine'] = train_data['earliesCreditLine'].apply(lambda x: int(x[-4:]))
        test_data['earliesCreditLine'] = test_data['earliesCreditLine'].apply(lambda x: int(x[-4:]))
        # print(train_data['earliesCreditLine'].value_counts(dropna=False).sort_index())
        train_data['grade'] = train_data['grade'].map({'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7})
        test_data['grade'] = test_data['grade'].map({'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7})
        # print(train_data['grade'].value_counts(dropna=False).sort_index())

        train_data['employmentLength'] = train_data['employmentLength'].fillna(
            train_data['employmentLength'].mode().values[0])
        test_data['employmentLength'] = test_data['employmentLength'].fillna(
            train_data['employmentLength'].mode().values[0])

        print('mode', train_data['employmentLength'].mode())

        print(train_data.isnull().any().sum())
        print(test_data.isnull().any().sum())

        missing = train_data.isnull().sum() / len(train_data)
        missing = missing[missing > 0]
        missing.sort_values(inplace=True)

        print(missing)

        # 异常值处理
        # all_data = all_data.copy()
        for fea in numerical_fea:
            train_data = find_outliers_by_3segama(train_data, fea)
            # print(all_data[fea+'_outliers'].value_counts())

        # get_dummies one hot 模式生成数据
        one_hot_columns = ['subGrade', 'homeOwnership', 'verificationStatus', 'purpose', 'regionCode']
        train_data = pd.get_dummies(train_data, columns=one_hot_columns, drop_first=True)
        test_data = pd.get_dummies(test_data, columns=one_hot_columns, drop_first=True)

        train_data['loanAmnt'] = np.floor_divide(train_data['loanAmnt'], 1000)
        test_data['loanAmnt'] = np.floor_divide(test_data['loanAmnt'], 1000)

        # 删除异常值
        for fea in numerical_fea:
            train_data = train_data[train_data[fea + '_outliers'] == '正常值']
            train_data = train_data.reset_index(drop=True)

        # 删除异常值标志列
        for tit in list(train_data.columns):
            if 'outliers' in tit:
                train_data.pop(tit)

        # label encode
        for col in tqdm(['employmentTitle', 'postCode', 'title']):
            le = LabelEncoder()
            le.fit(list(train_data[col].astype(str).values) + list(test_data[col].astype(str).values))
            train_data[col] = le.transform(list(train_data[col].astype(str).values))
            test_data[col] = le.transform(list(test_data[col].astype(str).values))

        # y值
        train_y = train_data.pop('isDefault')

        print(train_data.shape)
        print(test_data.shape)
        print(train_y.shape)

        # np.save(os.path.join(daikuan_path, 'train_data_stand'), train_data.values)
        # np.save(os.path.join(daikuan_path, 'test_data_stand'), test_data.values)
        # np.save(os.path.join(daikuan_path, 'train_y_stand'), train_y.values)

    @running_of_time
    def balance_data(self, train_x, train_y):
        # print('balance data')
        # print('count 1', train_y.tolist().count([1]))
        # print('count 0', train_y.tolist().count([0]))

        count0 = train_y.tolist().count([0])
        count1 = train_y.tolist().count([1])

        global source_x_dim
        source_x_dim = train_x.shape[1]

        tmp_x0 = np.zeros(shape=(count0, source_x_dim))
        tmp_y0 = np.zeros(shape=(count0, 1))

        tmp_x1 = np.zeros(shape=(count1, source_x_dim))
        tmp_y1 = np.zeros(shape=(count1, 1))

        idx0, idx1 = 0, 0
        for i in range(train_y.shape[0]):
            if train_y[i][0] == 0:
                tmp_x0[idx0, :] = train_x[i, :]
                tmp_y0[idx0, :] = train_y[i, :]
                idx0 += 1
            elif train_y[i][0] == 1:
                tmp_x1[idx1, :] = train_x[i, :]
                tmp_y1[idx1, :] = train_y[i, :]
                idx1 += 1

        location0 = np.random.permutation(tmp_x0.shape[0])
        location1 = np.random.permutation(tmp_x1.shape[0])
        tmp_x = np.concatenate([tmp_x0[location0][:samples], tmp_x1[location1][:samples]])
        tmp_y = np.concatenate([tmp_y0[location0][:samples], tmp_y1[location1][:samples]])
        location2 = np.random.permutation(tmp_x.shape[0])

        return tmp_x[location2], tmp_y[location2]

    def load_train_x_train_y_test_x(self, balance, s=True):
        if not s:
            train_x = np.load(os.path.join(daikuan_path, 'train_data_stand.npy'), allow_pickle=True)
            train_y = np.load(os.path.join(daikuan_path, 'train_y_stand.npy'), allow_pickle=True).reshape(-1, 1)
            test_x = np.load(os.path.join(daikuan_path, 'test_data_stand.npy'), allow_pickle=True)
        else:
            train_x = np.load(os.path.join(daikuan_path, 'train_data_stand_s.npy'), allow_pickle=True)
            train_y = np.load(os.path.join(daikuan_path, 'train_y_stand.npy'), allow_pickle=True).reshape(-1, 1)
            test_x = np.load(os.path.join(daikuan_path, 'test_data_stand_s.npy'), allow_pickle=True)

        print('train_x shape', train_x.shape, '\ntrain_y shape', train_y.shape, '\ntest_x shape', test_x.shape)
        if balance:
            train_x, train_y = self.balance_data(train_x, train_y)
        return train_x, train_y, test_x

    def shuffle(self, train_x, train_y):
        location = np.random.permutation(train_x.shape[0])
        return train_x[location], train_y[location]

    @running_of_time
    def train(self, model='lr', balance=False):
        """
        :return:
        """
        train_x, train_y, test = self.load_train_x_train_y_test_x(balance)
        train_xx, test_xx, train_yy, test_yy = train_test_split(train_x, train_y, train_size=0.9)
        # print('train_x shape', train_x.shape, '\ntrain_y shape', train_y.shape, '\ntest shape', test.shape)
        if model == 'lr':
            for _ in range(100, 301, 100):
                lr = LogisticRegression(penalty='l2', solver='liblinear', C=1, verbose=1, max_iter=_)
                lr.fit(train_xx, train_yy.reshape(-1, ))
                prob = lr.predict_proba(test_xx)
                i, j, k = metrics.roc_curve(test_yy, prob[:, 1])
                roc_auc = metrics.auc(i, j)
                print('lr auc', roc_auc)
                score = lr.score(test_xx, test_yy)
                joblib.dump(lr, os.path.join(daikuan_path,
                                             'lr_model_l2_{}_time_{}_score_{:.4f}_auc_{:.4f}'.format(
                                                 _, int(time.time()), score, roc_auc)))
                print('lr score', score)

            # r = lr.predict(test)
            # with open(os.path.join(daikuan_path, 'samples.csv'), mode='w') as f:
            #     f.write('id,isDefault\n')
            #     for idx, y in enumerate(r):
            #         print('{},{}'.format(idx + 800000, y))
            #         f.write('{},{}\n'.format(idx + 800000, y))
        elif model == 'rf':
            """
            cv rf best param {'max_depth': 12, 'max_features': 9, 'min_samples_leaf': 10, 'min_samples_split': 120}
            cv rf best socre 0.7136508939999999
            function:train, cost time: 299799.0638449192
            """
            param_test2 = {'max_depth': range(4, 13, 1), 'min_samples_split': range(80, 150, 20),
                           'min_samples_leaf': range(10, 60, 10), 'max_features': range(3, 11, 2)}
            # {'max_depth': range(4, 13, 1), 'min_samples_split': range(50, 201, 20)}
            # {'min_samples_split':range(80,150,20), 'min_samples_leaf':range(10,60,10)}
            # {'max_features':range(3,11,2)}
            m = GridSearchCV(estimator=RandomForestClassifier(oob_score=True,
                                                              verbose=1,
                                                              n_estimators=500,
                                                              max_depth=12,
                                                              min_samples_split=120,
                                                              min_samples_leaf=10,
                                                              max_features=9,
                                                              random_state=1),
                             param_grid=param_test2, scoring='roc_auc', iid=False, cv=10, n_jobs=4)
            m.fit(train_x, train_y.reshape(-1, ))
            # joblib.dump(m, os.path.join(daikuan_path, 'rf_model_{}'.format(int(time.time()))))
            print('cv rf best param', m.best_params_)
            print('cv rf best socre', m.best_score_)
        elif model == 'lgbm':
            # 网格搜索确认超参
            params = {
                # 'n_estimators': range(500, 4001, 500),
                'num_leaves': range(32, 64, 4),  # 细调  7
                'max_depth': range(1, 16, 2),  # 3
                # 'min_data_in_leaf': range(16, 128, 8),
                # 'min_child_weight': [0.001, 0.01, 0.1, 1, 1.5, 5, 10],
                # 'bagging_fraction': [i / 100 for i in range(50, 100, 1)],
                # 'bagging_freq': range(50, 100, 1),
                # 'feature_fraction': [i / 100 for i in range(50, 100, 1)],
                # 'reg_lambda': [0, 0.001, 0.01, 0.03, 0.08, 0.3, 0.5, 1, 1.5, 3, 9, 16],
                # 'reg_alpha': [0, 0.001, 0.01, 0.03, 0.08, 0.3, 0.5, 1, 1.5, 3, 9, 16],
                # 'min_split_gain': [i / 10 for i in range(0, 11)]
            }
            cv_fold = StratifiedKFold(n_splits=10, random_state=2000, shuffle=True)
            lgb_model = LGBMClassifier(
                n_estimators=2000,
                learning_rate=0.1,
                # num_leaves=7,
                # max_depth=3,
                # min_data_in_leaf=22,
                # min_child_weight=0,
                # bagging_fraction=0.9,
                # feature_fraction=0.7,
                # bagging_freq=30,
                # min_split_gain=0.9,
                # reg_lambda=0.08,
                # reg_alpha=0.01
            )
            grid_search = GridSearchCV(estimator=lgb_model, cv=cv_fold, param_grid=params, scoring='roc_auc', n_jobs=4,
                                       verbose=1)
            grid_search.fit(train_xx, train_yy.reshape(-1, ))
            print('cv lgbm best param', grid_search.best_params_)
            print('cv lgbm best socre', grid_search.best_score_)
            print('mean_test_score', grid_search.cv_results_['mean_test_score'])
            print('params', grid_search.cv_results_['params'])

            # 设置较小学习率来确定最终的num_boost_round
            # final_params = {
            #     'boosting_type': 'gbdt',
            #     'learning_rate': 0.01,
            #     'num_leaves': 7,
            #     'max_depth': 3,
            #     'min_data_in_leaf': 22,
            #     'min_child_weight': 0,
            #     'bagging_fraction': 0.9,
            #     'feature_fraction': 0.7,
            #     'bagging_freq': 30,
            #     'min_split_gain': 0.9,
            #     'reg_lambda': 0.08,
            #     'reg_alpha': 0.01
            # }

            # final_params = {
            #     'boosting_type': 'gbdt',
            #     'learning_rate': 0.01,
            #     'metric': 'auc',
            #     'num_leaves': 14,
            #     'max_depth': 19,
            #     'min_data_in_leaf': 37,
            #     'min_child_weight': 1.6,
            #     'feature_fraction': 0.69,
            #     'bagging_fraction': 0.98,
            #     'bagging_freq': 96,
            #     'min_split_gain': 0.4,
            #     'reg_lambda': 9,
            #     'reg_alpha': 7,
            # }
            # lgb_train = lgb.Dataset(train_x, train_y.reshape(-1, ))
            # cv_result = lgb.cv(train_set=lgb_train,
            #                    early_stopping_rounds=20,
            #                    num_boost_round=5000,
            #                    nfold=10,
            #                    stratified=True,
            #                    shuffle=True,
            #                    params=final_params,
            #                    metrics='auc',
            #                    seed=0,
            #                    )
            #
            # print('迭代次数{}'.format(len(cv_result['auc-mean'])))
            # print('交叉验证的AUC为{}'.format(max(cv_result['auc-mean'])))
            # 迭代次数3356
            # 交叉验证的AUC为0.727471435

            # acc_list = []
            # auc_list = []
            # for i in range(10):
            #     train_xx, test_xx, train_yy, test_yy = train_test_split(train_x, train_y, train_size=0.8)
            #     m = LGBMClassifier(
            #         n_estimators=581,
            #         learning_rate=0.1,
            #         num_leaves=7,
            #         max_depth=3,
            #         bagging_fraction=1.0,
            #         feature_fraction=1.0,
            #         bagging_freq=0,
            #         min_data_in_leaf=20,
            #         min_child_weight=0.001,
            #         min_split_gain=0,
            #         reg_lambda=0,
            #         reg_alpha=0,
            #         silent=True
            #     )
            #
            #     m.fit(train_xx, train_yy.reshape(-1, ))
            #     m.predict_proba(test_xx)
            #
            #     y_pred = m.predict(test_xx)
            #     y_predprob = m.predict_proba(test_xx)[:, 1]
            #     print('accuracy', metrics.accuracy_score(test_yy, y_pred))
            #     print('AUC', metrics.roc_auc_score(test_yy, y_predprob))
            #     acc_list.append(metrics.accuracy_score(test_yy, y_pred))
            #     auc_list.append(metrics.roc_auc_score(test_yy, y_predprob))
            # print('accuracy list', acc_list, sum(acc_list) / len(acc_list))
            # print('auc list', auc_list, sum(auc_list) / len(auc_list))
        elif model == 'gbdt':
            # 调参 https://blog.csdn.net/weixin_40924580/article/details/85043801
            # param_test1 = {'n_estimators': range(128, 256, 32)}
            param_test2 = {'max_depth': range(3, 14, 2), 'min_samples_split': range(100, 801, 200)}
            g_search = GridSearchCV(estimator=GradientBoostingClassifier(learning_rate=0.1, min_samples_split=300,
                                                                         min_samples_leaf=20,
                                                                         max_features='sqrt', subsample=0.8,
                                                                         random_state=10, verbose=1,
                                                                         n_estimators=192),
                                    param_grid=param_test2, scoring='roc_auc', iid=False, cv=5, verbose=1)
            # g_search.fit(train_xx, train_yy.reshape(-1, ))
            # print(g_search.best_params_)
            # print(g_search.best_score_)

            gbdt = GradientBoostingClassifier(n_estimators=192, learning_rate=0.1, min_samples_split=300,
                                              min_samples_leaf=20, verbose=1)
            gbdt.fit(train_xx, train_yy.reshape(-1, ))

            # gbdt = joblib.load(os.path.join(daikuan_path, 'gbdt_model_time_1599825531_score_0.6515666666666666'))

            y_pred = gbdt.predict(test_xx)
            y_predprob = gbdt.predict_proba(test_xx)[:, 1]
            print('accuracy', metrics.accuracy_score(test_yy, y_pred))
            print('AUC', metrics.roc_auc_score(test_yy, y_predprob))

            score = gbdt.score(test_xx, test_yy)
            joblib.dump(gbdt, os.path.join(daikuan_path, 'gbdt_model_time_{}_score_{}'.format(int(time.time()), score)))
            print('gbdt score', score)
        elif model == 'lgb':
            # https://blog.csdn.net/weiyongle1996/article/details/78446244  lgb boost不同参数
            train_m = lgb.Dataset(train_xx, train_yy.reshape(-1, ))
            test_m = lgb.Dataset(test_xx, test_yy.reshape(-1, ))

            cv_score = []
            kf = KFold(n_splits=10, shuffle=True, random_state=100)
            for i, (train_index, valid_index) in enumerate(kf.split(train_x, train_y)):
                train_x_split, train_y_split, test_x_split, test_y_split = train_x[train_index], train_y[train_index], \
                                                                           train_x[valid_index], train_y[valid_index]
                train_m = lgb.Dataset(train_x_split, train_y_split.reshape(-1, ))
                test_m = lgb.Dataset(test_x_split, test_y_split.reshape(-1, ))
                params = {
                    'boosting_type': 'gbdt',
                    'objective': 'binary',
                    'learning_rate': 0.01,
                    'metric': 'auc',
                    'num_leaves': 14,
                    'max_depth': 19,
                    'min_data_in_leaf': 37,
                    'min_child_weight': 1.6,
                    'reg_lambda': 9,
                    'reg_alpha': 7,
                    'feature_fraction': 0.69,
                    'bagging_fraction': 0.98,
                    'bagging_freq': 96,
                    'min_split_gain': 0.4,
                    'nthread': 8
                }
                model = lgb.train(params=params, train_set=train_m, valid_sets=test_m, num_boost_round=20000,
                                  verbose_eval=1000, early_stopping_rounds=200)
                val_pred = model.predict(test_x_split, num_iteration=model.best_iteration)
                cv_score.append(metrics.roc_auc_score(test_y_split, val_pred))

            print('cv scores', cv_score)
            print('cv score means', np.mean(cv_score))
            print('cv std', np.std(cv_score))
            #
            # val_pre_lgb = model.predict(test_xx)
            #
            # print(val_pre_lgb.shape)
            # print(test_yy.shape)
            # fpr, tpr, threshold = metrics.roc_curve(test_yy, val_pre_lgb)
            # print(fpr.shape, tpr.shape, threshold.shape)
            # roc_auc = metrics.auc(fpr, tpr)
            # joblib.dump(model,
            #             os.path.join(daikuan_path, 'lgb_model_time_{}_auc_{:.4f}'.format(int(time.time()), roc_auc)))
            # print('调参lightgbm单模型在验证集上的AUC：{}'.format(roc_auc))
            #
            # r = model.predict(test)
            # with open(os.path.join(daikuan_path, 'samples_gbdt20200916.csv'), mode='w') as f:
            #     f.write('id,isDefault\n')
            #     for idx, y in enumerate(r):
            #         f.write('{},{}\n'.format(idx + 800000, y))
        elif model == 'bayes':
            def rf_cv_lgb(num_leaves, max_depth, bagging_fraction, feature_fraction, bagging_freq, min_data_in_leaf,
                          min_child_weight, min_split_gain, reg_lambda, reg_alpha):
                # 建立模型
                model_lgb = lgb.LGBMClassifier(boosting_type='gbdt', objective='binary', metric='auc',
                                               learning_rate=0.1, n_estimators=2000,
                                               num_leaves=int(num_leaves), max_depth=int(max_depth),
                                               bagging_fraction=round(bagging_fraction, 2),
                                               feature_fraction=round(feature_fraction, 2),
                                               bagging_freq=int(bagging_freq), min_data_in_leaf=int(min_data_in_leaf),
                                               min_child_weight=min_child_weight, min_split_gain=min_split_gain,
                                               reg_lambda=reg_lambda, reg_alpha=reg_alpha,
                                               n_jobs=1
                                               )

                val = cross_val_score(model_lgb, train_x, train_y.reshape(-1, ), cv=10, scoring='roc_auc',
                                      n_jobs=4).mean()
                return val

            bayes_lgb = BayesianOptimization(
                rf_cv_lgb,
                {
                    'num_leaves': (32, 128),
                    'max_depth': (3, 25),
                    'bagging_fraction': (0.5, 1.0),
                    'feature_fraction': (0.5, 1.0),
                    'bagging_freq': (50, 200),
                    'min_data_in_leaf': (16, 128),
                    'min_child_weight': (0, 5),
                    'min_split_gain': (0.0, 1.0),
                    'reg_alpha': (0.0, 10),
                    'reg_lambda': (0.0, 10),
                }
            )
            bayes_lgb.maximize(n_iter=100)
            # joblib.dump(bayes_lgb, os.path.join(daikuan_path, 'bayes_lgb_model'))
            print(bayes_lgb.max)
            # {'target': 0.7327887126883773, 'params': {'bagging_fraction': 0.849424590455381, 'bagging_freq': 57.996644625593916, 'feature_fraction': 0.5738250741307653, 'max_depth': 3.381120810006032, 'min_child_weight': 4.8850064161708655, 'min_data_in_leaf': 24.766797838434663, 'min_split_gain': 0.3928718222398315, 'num_leaves': 40.12701452812764, 'reg_alpha': 9.72234441466076, 'reg_lambda': 4.566310059441557}}
        elif model == 'nn':
            def callb():
                callback = ModelCheckpoint(
                    filepath=os.path.join(corpus_root_path, 'daikuan', 'ann', 'ann_{val_acc:.4f}.model'),
                    monitor='roc_auc',
                    verbose=1, save_best_only=True, mode='max', period=2)
                earlystop = EarlyStopping(monitor='roc_auc', patience=2, verbose=1, mode='max')
                return [callback, earlystop]

            inputs = layers.Input(shape=(144,))
            dense2 = layers.Dense(64, activation='relu')(inputs)
            dropout1 = layers.Dropout(0.2)(dense2)
            dense3 = layers.Dense(32, activation='relu')(dropout1)
            dropout2 = layers.Dropout(0.2)(dense3)
            dense4 = layers.Dense(1, activation='sigmoid')(dropout2)
            m = models.Model(inputs, dense4)
            print(m.summary())
            m.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
            m.fit(train_xx, train_yy, batch_size=32, epochs=1, validation_split=0.1, callbacks=callb())
            y_pred = m.predict(test_xx)
            print(y_pred)
            print((y_pred == test_yy).tolist().count(True))

    @running_of_time
    def feature_select(self):
        from sklearn.feature_selection import SelectFromModel, VarianceThreshold, RFE
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.tree import DecisionTreeClassifier
        train_x, train_y, test = self.load_train_x_train_y_test_x(False)
        # model = DecisionTreeClassifier()
        # rfe = RFE(model, 64)
        # rfe = rfe.fit(train_x, train_y.reshape(-1, ))
        # print(rfe.support_)
        # print(rfe.ranking_)
        # print(VarianceThreshold(threshold=0).fit_transform(train_x).shape)
        print(SelectFromModel(GradientBoostingClassifier()).fit(train_x, train_y.reshape(-1, )))

    @running_of_time
    def train_model(self, model='lr', balance=False):
        kfold = KFold(n_splits=10, shuffle=True, random_state=2021)
        for epoch in range(1, 3):
            train_x, train_y, test = self.load_train_x_train_y_test_x(balance)
            train_x, train_y = self.shuffle(train_x, train_y)
            # train_x, train_y = self.balance_data(train_x, train_y)
            for i, (train_idx, valid_idx) in enumerate(kfold.split(train_x, train_y)):
                train_xx, train_yy = train_x[train_idx], train_y[train_idx]
                valid_xx, valid_yy = train_x[valid_idx], train_y[valid_idx]
                if model == 'lr':
                    lr = LogisticRegression(C=10, solver='liblinear', max_iter=100, n_jobs=1)
                    lr.fit(X=train_xx, y=train_yy.reshape(-1, ))
                    joblib.dump(lr, os.path.join(daikuan_path, 'model', 'lr_epoch_{}_k_{}.model'.format(epoch, i)))
                    # 验证集预测
                    prob = lr.predict_proba(valid_xx)
                    i, j, k = metrics.roc_curve(valid_yy, prob[:, 1])
                    roc_auc = metrics.auc(i, j)
                    print('lr valid roc_auc is', roc_auc)
                elif model == 'svm':
                    svc = SVC(C=10, kernel='rbf', verbose=True, max_iter=-1, gamma='scale')
                    svc.fit(train_xx, train_yy.reshape(-1, ))
                    joblib.dump(svc, os.path.join(daikuan_path, 'model', 'svc_epoch_{}_k_{}.model'.format(epoch, i)))
                    prob = svc.predict_proba(valid_xx)
                    i, j, k = metrics.roc_curve(valid_yy, prob[:, 1])
                    roc_auc = metrics.auc(i, j)
                    print('svc valid roc_auc is', roc_auc)
                elif model == 'lgb':
                    train_m = lgb.Dataset(train_xx, train_yy.reshape(-1, ))
                    valid_m = lgb.Dataset(valid_xx, valid_yy.reshape(-1, ))

                    params = {
                        'boosting_type': 'gbdt',
                        'objective': 'binary',
                        'learning_rate': 0.01,
                        'metric': 'auc',
                        'num_leaves': 14,
                        'max_depth': 19,
                        'min_data_in_leaf': 37,
                        'min_child_weight': 1.6,
                        'reg_lambda': 9,
                        'reg_alpha': 7,
                        'feature_fraction': 0.69,
                        'bagging_fraction': 0.98,
                        'bagging_freq': 96,
                        'min_split_gain': 0.4,
                        'nthread': 4
                    }

                    # params = {
                    #     'boosting_type': 'gbdt',
                    #     'objective': 'binary',
                    #     'learning_rate': 0.01,
                    #     'metric': 'auc',
                    #     'num_leaves': 32,
                    #     'max_depth': 6,
                    #     'min_data_in_leaf': 16,
                    #     'min_child_weight': 1.9,
                    #     # 'min_child_weight': 4.9,
                    #     'reg_lambda': 9,
                    #     'reg_alpha': 7,
                    #     'feature_fraction': 0.8,
                    #     'bagging_fraction': 0.65,
                    #     'bagging_freq': 50,
                    #     'min_split_gain': 0.4
                    # }
                    m = lgb.train(params=params, train_set=train_m, valid_sets=valid_m, num_boost_round=20000,
                                  verbose_eval=1000, early_stopping_rounds=200)
                    val_pre_lgb = m.predict(valid_xx)

                    fpr, tpr, threshold = metrics.roc_curve(valid_yy, val_pre_lgb)
                    print(fpr.shape, tpr.shape, threshold.shape)
                    roc_auc = metrics.auc(fpr, tpr)
                    joblib.dump(m, os.path.join(daikuan_path, 'model', 'lgb4_s_epoch_{}_k_{}.model'.format(epoch, i)))
                    print('调参lightgbm单模型在验证集上的AUC：{}'.format(roc_auc))
                elif model == 'lgbm':
                    model_lgb = lgb.LGBMClassifier(
                        boosting_type='gbdt', objective='binary', metric='auc',
                        learning_rate=0.1, n_estimators=2000,
                        num_leaves=40, max_depth=4,
                        bagging_fraction=0.85,
                        feature_fraction=0.57,
                        bagging_freq=58,
                        min_data_in_leaf=25,
                        min_child_weight=4.9, min_split_gain=0.4,
                        reg_lambda=4.6, reg_alpha=9.7,
                        n_jobs=4
                    )
                    model_lgb.fit(train_xx, train_yy)
                    from sklearn.metrics import roc_auc_score
                    from sklearn.model_selection import cross_validate
                    c = cross_validate(model_lgb, train_x, train_y, cv=10)
                    print('c', c)
                    return

    @running_of_time
    def predict(self):
        train_x, train_y, test = self.load_train_x_train_y_test_x(False)
        epochs = 3
        k = 10
        lr, lgb = [], []
        cv_score = []
        for epoch in range(epochs):
            for i in range(k):
                now = time.time()
                print('epochs', epoch, 'k', i)
                # model = joblib.load(os.path.join(daikuan_path, 'model', 'lr_epoch_{}_k_{}.model'.format(epoch, i)))
                # print('begin lr predict proba')
                # train_prob = model.predict_proba(test)
                # train_prob = train_prob[:, 1]
                # print('lr predict shape', train_prob.shape)
                # lr.append(train_prob.reshape(-1, 1))

                model = joblib.load(os.path.join(daikuan_path, 'model', 'lgb4_s_epoch_{}_k_{}.model'.format(epoch, i)))
                print('begin lgb predict')
                train_prob = model.predict(test)
                print('lgb predict shape', train_prob.shape)
                lgb.append(train_prob.reshape(-1, 1))
                print('epoch {}, k {} done. time: {}'.format(epoch, i, int(time.time() - now)))

        # merge_lr = np.concatenate(lr, axis=-1)
        # print('合并后的 lr shape', merge_lr.shape)
        # with open(os.path.join(daikuan_path, 'lr_train_kfold_5.npy'), mode='wb') as f:
        #     pickle.dump(merge_lr, f)
        merge_lgb = np.concatenate(lgb, axis=-1)
        print('合并后的 lgb shape', merge_lgb.shape)
        with open(os.path.join(daikuan_path, 'lgb4_s_test_kfold_10.npy'), mode='wb') as f:
            pickle.dump(merge_lgb, f)

    @running_of_time
    def test(self):
        # with open(os.path.join(daikuan_path, 'lr_train_kfold_5.npy'), mode='rb') as f:
        #     merge_lr = pickle.load(f)

        with open(os.path.join(daikuan_path, 'lgb4_s_test_kfold_10.npy'), mode='rb') as f:
            merge_lgb = pickle.load(f)

        # merge = np.concatenate([merge_lr, merge_lgb], axis=-1)

        merge = merge_lgb

        with open(os.path.join(daikuan_path, 'sample_submit6.csv'), mode='w') as f:
            f.write('id,isDefault\n')
            for idx, y in enumerate(np.mean(merge, axis=1)):
                f.write('{},{}\n'.format(idx + 800000, y))


if __name__ == '__main__':
    dai = DaiKuan()
    # dai.feature_select()
    # dai.format_train_x_train_y_test_x2(daikuan_classifier_path, daikuan_test_path)
    # dai.train_model(model='lgb', balance=False)
    # dai.train(model='bayes', balance=False)
    # dai.predict()
    dai.test()
