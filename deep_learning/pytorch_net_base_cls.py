import os
import sys
import torch.nn as nn
import torch
from torch import optim
import numpy as np

from tools import running_of_time


def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print({'Total': total_num, 'Trainable': trainable_num})


def data_iter(train_x, train_y, batch_size=32):
    i, llen = 0, len(train_x)
    location = np.random.permutation(llen)
    train_x, train_y = train_x[location], train_y[location]
    while i < llen:
        yield train_x[i:i + batch_size], train_y[i:i + batch_size]
        i += batch_size


def numpy2tensor(n):
    return torch.autograd.Variable(torch.from_numpy(n)).long()


def data2tensor(train_x, train_y, test_x, test_y):
    train_x = numpy2tensor(train_x)
    train_y = numpy2tensor(train_y).reshape(-1, )
    test_x = numpy2tensor(test_x)
    test_y = numpy2tensor(test_y).reshape(-1, )
    return train_x, train_y, test_x, test_y


@running_of_time
def train_model(net, train_x, train_y, epoch, lr, batch_size):
    print('begin training epoch', epoch)
    l_train = len(train_x)
    net.train()
    optimizer = optim.Adam(net.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    num_correct = 0
    for idx, (x, y) in enumerate(data_iter(train_x, train_y, batch_size)):
        optimizer.zero_grad()
        output = net(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()

        pred = output.argmax(dim=1)
        num_correct += torch.eq(pred, y).sum().float().item()
        if idx % 10 == 0:
            print('train epoch={}, samples={}->[{:.2%}], loss={:.6}'.format(epoch, idx * batch_size,
                                                                            idx * batch_size / l_train,
                                                                            loss.item() / batch_size))
    print('train epoch={}, correct={:.2%}'.format(epoch, num_correct / l_train))


def test_model(net, test_x, test_y):
    print('begin test')
    net.eval()
    correct = 0
    with torch.no_grad():
        for idx, (x, y) in enumerate(data_iter(test_x, test_y)):
            output = net(x)
            pred = output.argmax(dim=1)
            correct += torch.eq(pred, y).sum().float().item()
    print('test accuracy={:.2%}'.format(correct / len(test_x)))


class TrainTorch:
    def __init__(self, model):
        self.model = model

    def train(self, train_set, test_set, epochs, batch_size, lr, use_test=False):
        train_x, train_y = train_set
        test_x, test_y = test_set
        for epoch in range(epochs):
            print('begin training epoch at ', epoch)
            self.train_one_epoch(train_x, train_y, batch_size, lr)
            if use_test:
                self.evaluate(test_x, test_y)

    @running_of_time
    def train_one_epoch(self, train_x, train_y, batch_size, lr):
        l_train = len(train_x)
        self.model.train()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        num_correct = 0
        for idx, (x, y) in enumerate(data_iter(train_x, train_y, batch_size)):
            optimizer.zero_grad()
            output = self.model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()

            pred = output.argmax(dim=1)
            num_correct += torch.eq(pred, y).sum().float().item()
            if idx % 10 == 0:
                print('train samples={}->[{:.2%}], loss={:.6}'.format(idx * batch_size, idx * batch_size / l_train,
                                                                      loss.item() / batch_size))
        print('train correct={:.2%}'.format(num_correct / l_train))

    def evaluate(self, test_x, test_y):
        print('begin test')
        self.model.eval()
        correct = 0
        with torch.no_grad():
            for idx, (x, y) in enumerate(data_iter(test_x, test_y)):
                output = self.model(x)
                pred = output.argmax(dim=1)
                correct += torch.eq(pred, y).sum().float().item()
        print('test accuracy={:.2%}'.format(correct / len(test_x)))
