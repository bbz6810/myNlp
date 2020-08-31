import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.datasets import load_iris

x = np.array([[4, 7], [3.5, 8], [3.1, 6.2], [0.5, 1], [1, 2],
              [1.2, 1.9], [6, 2], [5.7, 1.5], [5.4, 2.2]])
y = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])


def test():
    x, y = load_iris(True)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=5)

    print(x_train.shape)
    print(y_train.shape)

    # classifier = SVC(kernel='linear')
    # classifier = SVC(kernel='poly', degree=3)
    classifier = SVC(kernel='rbf')
    classifier.fit(x_train, y_train)

    c = classifier.predict(x_test)
    print(c)
    print(y_test)


if __name__ == '__main__':
    test()
