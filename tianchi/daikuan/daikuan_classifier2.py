import numpy as np
import pandas as pd
import os
import time
import joblib
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from lightgbm import LGBMRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn import metrics

from tools import running_of_time
from corpus import daikuan_path

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

employment_length_dict = {'< 1 year': 11, '10+ years': 10, '8 years': 8, '1 year': 1, '2 years': 2, '5 years': 5,
                          '4 years': 4, '3 years': 3, '6 years': 6, '7 years': 7, '9 years': 9}

train_max_length = 800000
test_max_length = 800000
samples = 150000
source_x_dim = 959


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
        # y值
        train_y = train_data.pop('isDefault')

        all_data = pd.concat([train_data, test_data], ignore_index=True)

        all_data['employmentLength'] = all_data['employmentLength'].map(employment_length_dict)
        all_data.fillna(-1, inplace=True)

        print('begin one hot')
        # one hot 表示离散属性
        grade = np.array(all_data['grade']).reshape(-1, 1)
        grade_enc = OneHotEncoder().fit_transform(grade).toarray()
        all_data.pop('grade')

        sub_grade = np.array(all_data['subGrade']).reshape(-1, 1)
        sub_grade_enc = OneHotEncoder().fit_transform(sub_grade).toarray()
        all_data.pop('subGrade')

        employment_length = np.array(all_data['employmentLength']).reshape(-1, 1)
        employment_length_enc = OneHotEncoder(categories='auto').fit_transform(employment_length).toarray()
        all_data.pop('employmentLength')

        issue_date = np.array(all_data['issueDate']).reshape(-1, 1)
        issue_date_enc = OneHotEncoder().fit_transform(issue_date).toarray()
        all_data.pop('issueDate')

        earlies_credit_line = np.array(all_data['earliesCreditLine']).reshape(-1, 1)
        earlies_credit_line_enc = OneHotEncoder().fit_transform(earlies_credit_line).toarray()
        all_data.pop('earliesCreditLine')

        print('concat one hot train')
        all_data = pd.concat(
            [all_data, pd.DataFrame(grade_enc), pd.DataFrame(sub_grade_enc), pd.DataFrame(employment_length_enc),
             pd.DataFrame(issue_date_enc), pd.DataFrame(earlies_credit_line_enc)], axis=1)

        print('begin standard scaler')
        scaler = StandardScaler()
        for nor in normal:
            tmp_scale = scaler.fit(all_data[nor].values.reshape(-1, 1))
            all_data[nor] = scaler.fit_transform(all_data[nor].values.reshape(-1, 1), tmp_scale)

        print('begin pca')
        for i in [8, 16, 32]:
            print(i)
            pca = PCA(n_components=i)
            tmp = pca.fit_transform(all_data.values)
            np.save(os.path.join(daikuan_path, 'all_data_{}'.format(i)), tmp)
            np.save(os.path.join(daikuan_path, 'train_y_{}'.format(i)), train_y.values)

        # np.save(os.path.join(daikuan_path, 'all_data'), all_data.values)
        # np.save(os.path.join(daikuan_path, 'train_y'), train_y.values)

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
    def load_train_x_train_y_test_x(self):
        # all_data = np.load(os.path.join(daikuan_path, 'all_data.npy'))
        # train_y = np.load(os.path.join(daikuan_path, 'train_y.npy')).reshape(-1, 1)

        all_data = np.load(os.path.join(daikuan_path, 'all_data_32.npy'))
        train_y = np.load(os.path.join(daikuan_path, 'train_y_32.npy')).reshape(-1, 1)

        train_x = all_data[:train_max_length]
        test_x = all_data[train_max_length:]

        print('all_data shape', all_data.shape, '\ntrain_x shape', train_x.shape, '\ntrain_y shape', train_y.shape,
              '\ntest_x shape', test_x.shape)

        train_x, train_y = self.balance_data(train_x, train_y)
        return train_x, train_y, test_x

    @running_of_time
    def train(self, model='lr'):
        """
        lr score 0.80048125

        :return:
        """
        train_x, train_y, test = self.load_train_x_train_y_test_x()
        train_xx, test_xx, train_yy, test_yy = train_test_split(train_x, train_y, train_size=0.8)
        if model == 'lr':
            lr = LogisticRegression(penalty='l2', solver='liblinear', C=1, verbose=1)
            lr.fit(train_xx, train_yy.reshape(-1, ))
            score = lr.score(test_xx, test_yy)
            joblib.dump(lr, os.path.join(daikuan_path, 'lr_model_time_{}_score_{}'.format(int(time.time()), score)))
            print('lr score', score)

            # lr = joblib.load(os.path.join(daikuan_path, 'lr_model_time_1599821345_score_0.6554'))
            #
            # r = lr.predict(test)
            # with open(os.path.join(daikuan_path, 'samples.csv'), mode='w') as f:
            #     f.write('id,isDefault\n')
            #     for idx, y in enumerate(r):
            #         print('{},{}'.format(idx + 800000, y))
            #         f.write('{},{}\n'.format(idx + 800000, y))

            # r = lr.predict(test_xx)
            # for x, y in zip(r, test_yy):
            #     print(x, y)
            # print(lr.score(test_xx, test_yy))

        elif model == 'svm':
            # linear
            svc = SVC(C=1, kernel='rbf', verbose=True, max_iter=100)
            svc.fit(train_xx, train_yy.reshape(-1, ))
            score = svc.score(test_xx, test_yy)
            joblib.dump(svc, os.path.join(daikuan_path, 'svc_model_time_{}_score_{}'.format(int(time.time()), score)))
            print('svm score', score)

            # svc = joblib.load(os.path.join(daikuan_path, 'svc_model_time_1599796746_score_0.5245'))
            # r = svc.predict(test_xx)
        elif model == 'ada':
            ada = AdaBoostClassifier(DecisionTreeClassifier(max_depth=4), n_estimators=400, random_state=7)
            ada.fit(train_xx, train_yy.reshape(-1, ))
            score = ada.score(test_xx, test_yy)
            joblib.dump(ada, os.path.join(daikuan_path, 'ada_model_time_{}_score_{}'.format(int(time.time()), score)))
            print('ada score', score)

        elif model == 'rf':
            pass
        elif model == 'gbm':
            lgbm = LGBMRegressor(num_leaves=30
                                 , max_depth=5
                                 , learning_rate=.02
                                 , n_estimators=1000
                                 , subsample_for_bin=5000
                                 , min_child_samples=200
                                 , colsample_bytree=.2
                                 , reg_alpha=.1
                                 , reg_lambda=.1)
            lgbm.fit(train_xx, train_yy)
            score = lgbm.score(test_xx, test_yy)
            print('lgbm score', score)
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


if __name__ == '__main__':
    from tianchi.daikuan import DaiKuan
    dai = DaiKuan()
    # dai.format_train_x_train_y_test_x(daikuan_classifier_path, daikuan_test_path)
    # dai.load_train_x_train_y_test_x()
    dai.train(model='rf', balance=True)
