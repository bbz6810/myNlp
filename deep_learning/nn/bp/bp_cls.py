import numpy as np


class MulLayer:
    def __init__(self):
        self.x = None
        self.y = None

    def forward(self, x, y):
        self.x = x
        self.y = y
        out = x * y
        return out

    def backward(self, dout):
        dx = dout * self.y
        dy = dout * self.x
        return dx, dy


class AddLayer:
    def __init__(self):
        self.x = None
        self.y = None

    def forward(self, x, y):
        self.x = x
        self.y = y
        out = x + y
        return out

    def backward(self, dout):
        dx = dout * 1
        dy = dout * 1
        return dx, dy


class Relu:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0
        return out

    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout
        return dx


class Sigmoid:
    def __init__(self):
        self.out = None

    def forward(self, x):
        out = 1 / (1 + np.exp(-x))
        self.out = out
        return out

    def backward(self, dout):
        dx = dout * (1.0 - self.out) * self.out
        return dx


class Affine:
    def __init__(self, w, b):
        self.w = w
        self.b = b
        self.x = None
        self.dw = None
        self.db = None

    def forward(self, x):
        self.x = x
        out = np.dot(x, self.w) + self.b
        return out

    def backward(self, dout):
        dx = np.dot(dout, self.w.T)
        self.dw = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)
        return dx


def softmax(a):
    exp_a = np.exp(a)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y


def cross_entropy_error(y, t):
    delta = 1e-7
    return -np.sum(t * np.log(y + delta))


class SoftMaxWithLoss:
    def __init__(self):
        self.loss = None  # 损失
        self.y = None  # soft max的输出
        self.t = None  # 监督数据one-hot vector

    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)
        return self.loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        dx = (self.y - self.t) / batch_size
        return dx


def jh():
    a = np.array([[1, -2], [-3, 4]])
    relu = Relu()
    relu.forward(a)

    dout = np.ones(shape=(2, 2))
    relu.backward(dout)

    sigmoid = Sigmoid()
    sigmoid.forward(a)
    dout = 1
    sigmoid.backward(dout)


def mullayer():
    apple = 100
    apple_num = 2
    orange = 150
    orange_num = 3
    tax = 1.1

    mul_apple = MulLayer()
    mul_orange = MulLayer()
    add_apple_orange = AddLayer()
    mul_tax = MulLayer()

    apple_price = mul_apple.forward(apple, apple_num)
    orange_price = mul_orange.forward(orange, orange_num)
    all_price = add_apple_orange.forward(apple_price, orange_price)
    price = mul_tax.forward(all_price, tax)

    print(price)

    dprice = 1
    dall_price, dtax = mul_tax.backward(dprice)
    dapple_price, dorange_price = add_apple_orange.backward(dall_price)
    dapple, dapple_num = mul_apple.backward(dapple_price)
    dorange, dorange_num = mul_orange.backward(dorange_price)
    print(dapple, dapple_num, dorange, dorange_num, dtax)


if __name__ == '__main__':
    # mullayer()
    jh()
