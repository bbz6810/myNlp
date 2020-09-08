import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

iris = load_iris()
# print(iris.data)
x0 = iris.data
y = iris.target

x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, train_size=0.6, random_state=4)

standard = StandardScaler()
x1 = standard.fit_transform(iris.data)
# print(x)

minmax = MinMaxScaler()
x2 = minmax.fit_transform(iris.data)

# for i in range(iris.data.shape[0]):
#     print(iris.data[i], x1[i], x2[i])

# lr = LogisticRegression(penalty='l1', solver='liblinear', C=8, max_iter=1000)
# lr.fit(x_train, y_train)
# s = lr.score(x_test, y_test)
# print(s)

# normal = Normalizer()
# x3 = normal.fit_transform(iris.data)
#
#
# data_target_list = [(x0, y), (x1, y), (x2, y), (x3, y)]
#
# lr = LogisticRegression(penalty='l2', solver='liblinear', C=1)
#
# for data, target in data_target_list:
#     x_train, x_test, y_train, y_test = train_test_split(data, target, train_size=0.6, random_state=5)
#     lr.fit(x_train, y_train)
#     s = lr.score(x_test, y_test)
#     print(s)


# a = SelectFromModel(LogisticRegression(C=20)).fit_transform(x1, y)
# print(a)


pca = PCA(n_components=3).fit_transform(x0, y)
# print(pca)


lda = LDA(n_components=2).fit_transform(x0, y)
print(lda)