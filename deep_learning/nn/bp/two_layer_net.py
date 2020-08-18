import numpy as np
from collections import OrderedDict
from deep_learning.nn.bp.bp_cls import AddLayer, MulLayer, Relu, Sigmoid, SoftMaxWithLoss, Affine


class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        self.params = dict()
        self.params['w1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(shape=(hidden_size,))
        self.params['w2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(shape=(output_size,))

        self.layers = OrderedDict()
        self.layers['affine1'] = Affine(self.params['w1'], self.params['b1'])
        self.layers['relu1'] = Relu()
        self.layers['affine2'] = Affine(self.params['w2'], self.params['b2'])
        self.lastlayer = SoftMaxWithLoss()

    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)
        return x

    def loss(self, x, t):
        y = self.predict(x)
        return self.lastlayer.forward(y, t)

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        if t.ndim != 1:
            t = np.argmax(t, axis=1)
        accurncy = np.sum(y == t) / float(x.shape[0])
        return accurncy

    def gradient(self, x, t):
        self.loss(x, t)

        dout = 1
        dout = self.lastlayer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        # 设定
        grads = dict()
        grads['w1'] = self.layers['affine1'].dw
        grads['b1'] = self.layers['affine1'].db
        grads['w2'] = self.layers['affine2'].dw
        grads['b2'] = self.layers['affine2'].db

        return grads


def test():
    x = np.array([[1, 1], [2, 2], [3, 3], [4, 4]])
    y = np.array([[1], [1], [0], [0]])
    print(x)
    print(y)

    network = TwoLayerNet(2, 4, 1)

    for i in range(100):
        grad = network.gradient(x, y)

        for key in ['w1', 'b1', 'w2', 'b2']:
            network.params[key] -= 0.1 * grad[key]

        loss = network.loss(x, y)


if __name__ == '__main__':
    test()
