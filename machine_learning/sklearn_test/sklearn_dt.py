"""房屋价格预估

"""

import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn import datasets
from sklearn.metrics import mean_squared_error, explained_variance_score
from sklearn.utils import shuffle
import matplotlib.pyplot as plt


def test():
    house_data = datasets.load_boston()
    x, y = shuffle(house_data.data, house_data.target, random_state=7)
    num_training = int(0.8 * len(x))
    print(x.shape)
    print(y.shape)
    x_train, y_train = x[:num_training], y[:num_training]
    x_test, y_test = x[num_training:], y[num_training:]

    dt_regressor = DecisionTreeRegressor(max_depth=4)
    dt_regressor.fit(x_train, y_train)

    ad_regressor = AdaBoostRegressor(DecisionTreeRegressor(max_depth=4), n_estimators=400, random_state=7)
    ad_regressor.fit(x_train, y_train)

    y_dt_predict = dt_regressor.predict(x_test)
    mse = mean_squared_error(y_test, y_dt_predict)
    evs = explained_variance_score(y_test, y_dt_predict)
    print('dt', evs, mse)

    y_ad_predict = ad_regressor.predict(x_test)
    mse = mean_squared_error(y_test, y_ad_predict)
    evs = explained_variance_score(y_test, y_ad_predict)
    print('ad', evs, mse)

    plot_feature_importance(dt_regressor.feature_importances_, 'dt', house_data.feature_names)
    plot_feature_importance(ad_regressor.feature_importances_, 'ad', house_data.feature_names)


def plot_feature_importance(feature_importance, title, feature_name):
    # 将重要的性质标准化
    feature_importance = 100. * (feature_importance / max(feature_importance))
    # 将得分从高到低排序
    index_sorted = np.flipud(np.argsort(feature_importance))
    # 让x坐标轴上的标签居中
    pos = np.arange(index_sorted.shape[0]) + 0.5

    # 画图
    plt.figure()
    plt.bar(pos, feature_importance[index_sorted], align='center')
    plt.xticks(pos, feature_name[index_sorted])
    plt.ylabel('important')
    plt.title(title)
    plt.show()


if __name__ == '__main__':
    test()
