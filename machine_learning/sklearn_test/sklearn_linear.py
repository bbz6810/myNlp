import numpy as np
from sklearn import linear_model
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.metrics import explained_variance_score
from corpus import cat_data

x = np.array([[4, 7], [3.5, 8], [3.1, 6.2], [0.5, 1], [1, 2],
              [1.2, 1.9], [6, 2], [5.7, 1.5], [5.4, 2.2]])
y = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])


def plot_classifier(classifier, x, y):
    # 定义图形的取值范围
    x_min, x_max = min(x[:, 0]) - 1., max(x[:, 0]) + 1.
    y_min, y_max = min(x[:, 1]) - 1., max(x[:, 1]) + 1.

    # 设置网络格数据步长
    step_size = 0.01

    # 定义网络
    x_values, y_values = np.meshgrid(np.arange(x_min, x_max, step_size), np.arange(y_min, y_max, step_size))
    # 计算分类器输出结果
    mesh_output = classifier.predict(np.c_[x_values.ravel(), y_values.ravel()])
    # 数组维度变形
    mesh_output = mesh_output.reshape(x_values.shape)

    plt.figure()
    plt.pcolormesh(x_values, y_values, mesh_output, cmap=plt.cm.gray)
    plt.scatter(x[:, 0], x[:, 1], c=y, s=80, edgecolors='black', linewidths=1, cmap=plt.cm.Paired)
    # 设置图形取值范围
    plt.xlim(x_values.min(), x_values.max())
    plt.ylim(y_values.min(), y_values.max())

    # 设置x轴与y轴
    plt.xticks((np.arange(int(min(x[:, 0]) - 1), int(min(x[:, 0]) + 1), 1.0)))
    plt.xticks((np.arange(int(min(x[:, 1]) - 1), int(min(x[:, 1]) + 1), 1.0)))
    plt.show()


def test():
    # C 越大 惩罚权重越大，分解边越间隔最大化，分类边界越优
    classifier = linear_model.LogisticRegression(penalty='l2', solver='liblinear', C=100)
    classifier.fit(x, y)

    plot_classifier(classifier, x, y)


def bayse():
    classifier = GaussianNB()
    classifier.fit(x, y)
    y_pred = classifier.predict(x)
    plot_classifier(classifier, x, y)


def randomt():
    X = []
    with open(cat_data, 'r') as f:
        for line in f.readlines():
            data = line[:-1].split(',')
            X.append(data)
    X = np.array(X)

    label_encoder = []
    X_encoded = np.empty(X.shape)
    for i, item in enumerate(X[0]):
        label_encoder.append(preprocessing.LabelEncoder())
        X_encoded[:, i] = label_encoder[-1].fit_transform(X[:, i])
    X = X_encoded[:, :-1].astype(int)
    y = X_encoded[:, -1].astype(int)

    classifier = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=7)
    classifier.fit(X, y)
    _y = classifier.predict(X)
    evls = explained_variance_score(y, _y)
    print(evls)
    # plot_classifier(classifier, X, y)


if __name__ == '__main__':
    # test()
    # bayse()
    randomt()
