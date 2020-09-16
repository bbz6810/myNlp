import os
import time
import datetime
import joblib
import numpy as np
import pandas as pd
from keras import models, layers
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, GradientBoostingClassifier, \
    VotingClassifier, RandomForestClassifier
from lightgbm import LGBMRegressor, LGBMClassifier
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn import metrics
from sklearn.feature_selection import VarianceThreshold
from tqdm import tqdm
from bayes_opt import BayesianOptimization

from tools import running_of_time
from corpus import daikuan_classifier_path, daikuan_test_path, corpus_root_path, daikuan_path

"""
    1、删除错误列数据并返回x和y
    2、处理离散化数据
    3、处理连续数据
    4、拆分训练和测试数据，并入模型跑数据
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
samples = 119000
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
        print(numerical_fea)
        print(category_fea)
        numerical_fea.remove('isDefault')
        # print(numerical_fea)
        # # print(category_fea)
        # # 连续和离散型变量
        numerical_serial_fea = []  # 连续
        numerical_noserial_fea = []  # 离散
        for fea in numerical_fea:
            temp = train_data[fea].nunique()
            if temp <= 10:
                numerical_noserial_fea.append(fea)
            else:
                numerical_serial_fea.append(fea)
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

        np.save(os.path.join(daikuan_path, 'train_data_stand'), train_data.values)
        np.save(os.path.join(daikuan_path, 'test_data_stand'), test_data.values)
        np.save(os.path.join(daikuan_path, 'train_y_stand'), train_y.values)

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

    @running_of_time
    def load_train_x_train_y_test_x(self, balance):
        train_x = np.load(os.path.join(daikuan_path, 'train_data_stand.npy'), allow_pickle=True)
        train_y = np.load(os.path.join(daikuan_path, 'train_y_stand.npy'), allow_pickle=True).reshape(-1, 1)
        test_x = np.load(os.path.join(daikuan_path, 'test_data_stand.npy'), allow_pickle=True)

        print('train_x shape', train_x.shape, '\ntrain_y shape', train_y.shape, '\ntest_x shape', test_x.shape)
        if balance:
            train_x, train_y = self.balance_data(train_x, train_y)
        return train_x, train_y, test_x

    @running_of_time
    def train(self, model='lr', balance=False):
        """
        :return:
        """
        train_x, train_y, test = self.load_train_x_train_y_test_x(balance)
        train_xx, test_xx, train_yy, test_yy = train_test_split(train_x, train_y, train_size=0.8)
        print('train_x shape', train_x.shape, '\ntrain_y shape', train_y.shape, '\ntest shape', test.shape)
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
            param_test2 = {'min_samples_split': range(50, 100, 10), 'min_samples_leaf': range(5, 15, 2)}
            # {'max_depth': range(4, 13, 1), 'min_samples_split': range(50, 201, 20)}
            # {'min_samples_split':range(80,150,20), 'min_samples_leaf':range(10,60,10)}
            # {'max_features':range(3,11,2)}
            m = GridSearchCV(estimator=RandomForestClassifier(oob_score=True,
                                                              verbose=1,
                                                              n_estimators=256,
                                                              max_depth=19,
                                                              # min_samples_split=80,
                                                              # min_samples_leaf=10,
                                                              max_features='sqrt'),
                             param_grid=param_test2, scoring='roc_auc', iid=False, cv=5, n_jobs=4)
            m.fit(train_x, train_y.reshape(-1, ))
            # joblib.dump(m, os.path.join(daikuan_path, 'rf_model_{}'.format(int(time.time()))))
            print('cv rf best param', m.best_params_)
            print('cv rf best socre', m.best_score_)
        elif model == 'lgbm':
            params = {
                # 'boosting_type': ('gbdt', 'dart', 'rf'),
                # 'num_leaves': range(10, 80, 5),  # 细调
                # 'max_depth': range(3, 15, 2),
                # 'n_estimators': range(100, 400, 50),
                'min_split_gain': [i / 10 for i in range(0, 11)],
                # 'min_data_in_leaf': range(20, 51, 5),
                # 'bagging_fraction': [i / 10 for i in range(5, 10, 1)],
                # 'feature_fraction': [i / 10 for i in range(5, 10, 1)],
                # 'bagging_freq': range(0, 81, 10),
                # 'reg_lambda': [0, 0.001, 0.01, 0.03, 0.08, 0.3, 0.5],
                # 'reg_alpha': [0, 0.001, 0.01, 0.03, 0.08, 0.3, 0.5]
            }
            m = GridSearchCV(estimator=LGBMClassifier(
                objective='binary',
                boosting_type='gbdt',
                n_estimators=200,
                learning_rate=0.1,
                num_leaves=40,
                max_depth=13,
                bagging_fraction=0.9,
                feature_fraction=0.6,
                bagging_freq=40,
                min_data_in_leaf=20,
                min_child_weight=0.001,
                min_split_gain=0.7,
                reg_lambda=0.01,
                reg_alpha=0.5
            ), cv=10, param_grid=params, scoring='roc_auc', n_jobs=4, verbose=1)
            m.fit(train_xx, train_yy.reshape(-1, ))
            print('cv lgbm best param', m.best_params_)
            print('cv lgbm best socre', m.best_score_)

            # m = LGBMClassifier(
            #     n_estimators=128,
            #     learning_rate=0.1,
            #     num_leaves=32,
            #     max_depth=9,
            #     bagging_fraction=0.9,
            #     feature_fraction=0.6,
            #     bagging_freq=40,
            #     min_data_in_leaf=20,
            #     min_child_weight=0.001,
            #     min_split_gain=0.7,
            #     reg_lambda=0.01,
            #     reg_alpha=0.5
            # )
            # m.fit(train_xx, train_yy.reshape(-1, ))
            # joblib.dump(m, os.path.join(daikuan_path, 'lgbm_model_{}'.format(int(time.time()))))
            # m = joblib.load(os.path.join(daikuan_path, 'lgbm_model_1600173987'))
            # r = m.predict(test_xx)
            # for i in r:
            #     print(i)
            # print('cv lgbm best param', m.best_params_)
            # print('cv lgbm best socre', m.best_score_)

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

            # cv_score = []
            # kf = KFold(n_splits=5, shuffle=True, random_state=100)
            # for i, (train_index, valid_index) in enumerate(kf.split(train_x, train_y)):
            #     train_x_split, train_y_split, test_x_split, test_y_split = train_x[train_index], train_y[train_index], \
            #                                                                train_x[valid_index], train_y[valid_index]
            #     train_m = lgb.Dataset(train_x_split, train_y_split.reshape(-1, ))
            #     test_m = lgb.Dataset(test_x_split, test_y_split.reshape(-1, ))
            #     # params = {
            #     #     'boosting_type': 'gbdt',
            #     #     'objective': 'binary',
            #     #     'learning_rate': 0.01,
            #     #     'metric': 'auc',
            #     #     'num_leaves': 14,
            #     #     'max_depth': 19,
            #     #     'min_data_in_leaf': 37,
            #     #     'min_child_weight': 1.6,
            #     #     'reg_lambda': 9,
            #     #     'reg_alpha': 7,
            #     #     'feature_fraction': 0.69,
            #     #     'bagging_fraction': 0.98,
            #     #     'bagging_freq': 96,
            #     #     'min_split_gain': 0.4,
            #     #     'nthread': 8
            #     # }
            #     params = {
            #         'boosting_type': 'gbdt',
            #         'objective': 'binary',
            #         'learning_rate': 0.01,
            #         'metric': 'auc',
            #         'num_leaves': 15,
            #         'max_depth': 18,
            #         'min_data_in_leaf': 23,
            #         'min_child_weight': 1.4,
            #         'reg_lambda': 9,
            #         'reg_alpha': 8,
            #         'feature_fraction': 0.56,
            #         'bagging_fraction': 0.92,
            #         'bagging_freq': 99,
            #         'min_split_gain': 0.84,
            #         'nthread': 8
            #     }
            #     model = lgb.train(params=params, train_set=train_m, valid_sets=test_m, num_boost_round=20000,
            #                       verbose_eval=1000, early_stopping_rounds=200)
            #     val_pred = model.predict(test_x_split, num_iteration=model.best_iteration)
            #     cv_score.append(metrics.roc_auc_score(test_y_split, val_pred))
            #
            # print('cv scores', cv_score)
            # print('cv score means', np.mean(cv_score))
            # print('cv std', np.std(cv_score))

            # params = {
            #     'boosting_type': 'gbdt',
            #     'objective': 'binary',
            #     'learning_rate': 0.01,
            #     'metric': 'auc',
            #     'num_leaves': 15,
            #     'max_depth': 18,
            #     'min_data_in_leaf': 23,
            #     'min_child_weight': 1.4,
            #     'reg_lambda': 9,
            #     'reg_alpha': 8,
            #     'feature_fraction': 0.56,
            #     'bagging_fraction': 0.92,
            #     'bagging_freq': 99,
            #     'min_split_gain': 0.84,
            #     'nthread': 8
            # }

            # cv_result_lgb = lgb.cv(train_set=train_m,
            #                        early_stopping_rounds=1000,
            #                        num_boost_round=20000,
            #                        nfold=10,
            #                        stratified=True,
            #                        shuffle=True,
            #                        params=params,
            #                        metrics='auc',
            #                        seed=2020
            #                        )
            # print('迭代次数{}'.format(len(cv_result_lgb['auc-mean'])))
            # print('最终模型的AUC为{}'.format(max(cv_result_lgb['auc-mean'])))

            params = {
                'boosting_type': 'gbdt',
                'objective': 'binary',
                'learning_rate': 0.01,
                'n_estimators': 192,
                'metric': 'auc',
                'num_leaves': 15,
                'max_depth': 18,
                'min_data_in_leaf': 32,
                'min_child_weight': 1.4,
                'min_split_gain': 0.4,
                'reg_lambda': 9,
                'reg_alpha': 8,
                'feature_fraction': 0.6,
                'bagging_fraction': 0.95,
                'bagging_freq': 95,
                'nthread': 8
            }
            model = lgb.train(params, train_set=train_m, valid_sets=test_m, num_boost_round=10000, verbose_eval=1000,
                              early_stopping_rounds=200)

            val_pre_lgb = model.predict(test_xx)

            print(val_pre_lgb.shape)
            print(test_yy.shape)
            fpr, tpr, threshold = metrics.roc_curve(test_yy, val_pre_lgb)
            print(fpr.shape, tpr.shape, threshold.shape)
            roc_auc = metrics.auc(fpr, tpr)
            joblib.dump(model,
                        os.path.join(daikuan_path, 'lgb_model_time_{}_auc_{:.4f}'.format(int(time.time()), roc_auc)))
            print('调参lightgbm单模型在验证集上的AUC：{}'.format(roc_auc))

            r = model.predict(test)
            with open(os.path.join(daikuan_path, 'samples_gbdt20200916.csv'), mode='w') as f:
                f.write('id,isDefault\n')
                for idx, y in enumerate(r):
                    f.write('{},{}\n'.format(idx + 800000, y))

        elif model == 'bayes':
            def rf_cv_lgb(num_leaves, max_depth, bagging_fraction, feature_fraction, bagging_freq, min_data_in_leaf,
                          min_child_weight, min_split_gain, reg_lambda, reg_alpha):
                # 建立模型
                model_lgb = lgb.LGBMClassifier(boosting_type='gbdt', bjective='binary', metric='auc',
                                               learning_rate=0.1, n_estimators=5000,
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
                    'num_leaves': (10, 20),
                    'max_depth': (3, 25),
                    'bagging_fraction': (0.5, 1.0),
                    'feature_fraction': (0.5, 1.0),
                    'bagging_freq': (0, 100),
                    'min_data_in_leaf': (10, 100),
                    'min_child_weight': (0, 5),
                    'min_split_gain': (0.0, 1.0),
                    'reg_alpha': (0.0, 10),
                    'reg_lambda': (0.0, 10),
                }
            )
            bayes_lgb.maximize(n_iter=10)
            # joblib.dump(bayes_lgb, os.path.join(daikuan_path, 'bayes_lgb_model'))
            print(bayes_lgb.max)


if __name__ == '__main__':
    dai = DaiKuan()
    # dai.format_train_x_train_y_test_x(daikuan_classifier_path, daikuan_test_path)
    # dai.load_train_x_train_y_test_x()
    dai.train(model='lgb', balance=True)
