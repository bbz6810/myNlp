import numpy as np
import pandas as pd
import os
import time
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler

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

employment_length_dict = {'< 1 year': 11, '10+ years': 10, '8 years': 8, '1 year': 1, '2 years': 2, '5 years': 5,
                          '4 years': 4, '3 years': 3, '6 years': 6, '7 years': 7, '9 years': 9}

train_max_length = 800000
test_max_length = 200000


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

        all_data.to_csv(os.path.join(daikuan_path, 'all_data.csv'))
        train_y.to_csv(os.path.join(daikuan_path, 'train_y.csv'))
        #
        # train_x = all_data[:train_max_length]
        # test_x = all_data[test_max_length:]
        # print('all_data shape', all_data.shape, '\ntrain_x shape', train_x.shape, '\ntrain_y shape', train_y.shape,
        #       '\ntest_x shape', test_x.shape)
        #
        # return train_x, train_y, test_x

    @running_of_time
    def load_train_x_train_y_test_x(self):
        all_data = pd.read_csv(os.path.join(corpus_root_path, 'all_data.csv'))
        train_y = pd.read_csv(os.path.join(corpus_root_path, 'train_y.csv'))

        train_x = all_data[:train_max_length]
        test_x = all_data[train_max_length:]

        print('all_data shape', all_data.shape, '\ntrain_x shape', train_x.shape, '\ntrain_y shape', train_y.shape,
              '\ntest_x shape', test_x.shape)
        return train_x, train_y, test_x

    @running_of_time
    def train(self, model='lr'):
        """
        lr score 0.80048125

        :return:
        """
        train_x, train_y, test = self.load_train_x_train_y_test_x()
        train_xx, test_xx, train_yy, test_yy = train_test_split(train_x, train_y, train_size=0.6)
        if model == 'lr':
            # lr = LogisticRegression(penalty='l2', solver='sag', C=1, max_iter=1000, n_jobs=4, class_weight='balanced')
            # lr.fit(train_xx, train_yy)
            # score = lr.score(test_xx, test_yy)
            # joblib.dump(lr, os.path.join(daikuan_path, 'lr_model_time_{}_score_{}'.format(int(time.time()), score)))
            lr = joblib.load(os.path.join(daikuan_path, 'lr_model_time_1599796866_score_0.74336875'))
            r = lr.predict(test)
            for idx, y in enumerate(r):
                print('{},{}'.format(idx + 800000, y))
            # r = lr.predict(test_xx)
            # for x, y in zip(r, test_yy):
            #     print(x, y)
            # print(lr.score(test_xx, test_yy))
        elif model == 'svm':
            # linear
            svc = SVC(C=1, kernel='rbf', class_weight='balanced')
            svc.fit(train_xx, train_yy)
            score = svc.score(test_xx, test_yy)
            joblib.dump(svc, os.path.join(daikuan_path, 'svc_model_time_{}_score_{}'.format(int(time.time()), score)))

            # svc = joblib.load(os.path.join(daikuan_path, 'svc_model_time_1599796746_score_0.5245'))
            # # r = svc.predict(test_xx)
            #
            # r = svc.decision_function(test_xx)
            #
            # for x, y in zip(r, test_yy):
            #     print(x, y)
            # print(svc.score(test_xx, test_yy))
        elif model == 'gbdt':
            pass
        elif model == 'rf':
            pass


if __name__ == '__main__':
    dai = DaiKuan()
    dai.train(model='svm')
