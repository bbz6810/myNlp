import os
from functools import reduce
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from keras import models, layers
from sklearn.model_selection import KFold
from deep_learning.nn_param import NN
from deep_learning.data_pretreatment import Pretreatment
from tools import running_of_time
from corpus import corpus_root_path

batch_size = 32
epochs = 16


class FastText(NN):
    def __init__(self, nn_param):
        super(FastText, self).__init__(nn_param=nn_param)

    def model(self, embedding_matrix):
        model = models.Sequential()
        model.add(layers.Embedding(self.nn_param.vocab_size + 1, self.nn_param.embedding_dim,
                                   input_length=self.nn_param.max_words))
        # input_length=self.nn_param.max_words, weights=[embedding_matrix]))
        model.add(layers.GlobalAveragePooling1D())
        if self.nn_param.class_num == 2:
            model.add(layers.Dense(1, activation='sigmoid'))
            model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        else:
            model.add(layers.Dense(self.nn_param.class_num, activation='softmax'))
            model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        print(model.summary())
        return model

    def train(self, train_x, train_y, embedding_matrix):
        self.nn = self.model(embedding_matrix)
        self.nn.fit(train_x, train_y, epochs=epochs, batch_size=batch_size, validation_split=0.2)
        weight = self.nn.get_weights()
        for w in weight:
            print(w.shape)

    def predict(self, test_x, test_y):
        _y = self.nn.predict(test_x)
        loss, acc = self.nn.evaluate(test_x, test_y)
        print("精度", acc, '损失', loss)
        # self.score(_y, test_y)


class FastText2(nn.Module):
    def __init__(self, nn_param):
        super(FastText2, self).__init__()
        self.nn_param = nn_param
        print(nn_param.__dict__)
        self.embedding = nn.Embedding(self.nn_param.vocab_size + 1, self.nn_param.embedding_dim)
        self.embedding.weight.requires_grad = True
        self.fc = nn.Sequential(
            nn.Linear(nn_param.embedding_dim, 128),
            # nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Linear(128, nn_param.class_num)
        )

    def forward(self, x):
        x = self.embedding(x)
        out = self.fc(torch.mean(x, dim=1))
        return out


def data_iter(train_x, train_y):
    i = 0
    llen = len(train_x)
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
def train_model(net, train_x, train_y, epoch, lr):
    print('begin training epoch', epoch)
    l_train = len(train_x)
    net.train()
    optimizer = optim.Adam(net.parameters(), lr=lr)
    # optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    num_correct = 0
    for idx, (x, y) in enumerate(data_iter(train_x, train_y)):
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


def run_pytorch():
    pretreatment = Pretreatment()
    train_x, test_x, train_y, test_y = pretreatment.train_test_split(c=3, y_one_hot=False, test_size=0.1)  # 0.9785
    train_x, train_y, test_x, test_y = data2tensor(train_x, train_y, test_x, test_y)

    fasttext = FastText2(pretreatment.nnparam)
    p = 0
    for k, v in fasttext.named_parameters():
        print('name', k, 'param', v.size())
        p += reduce(lambda x, y: x * y, list(v.size()))
    print(fasttext)
    print('参数量', p)

    for epoch in range(epochs):
        train_model(net=fasttext, train_x=train_x, train_y=train_y, epoch=epoch, lr=0.0001)
        test_model(fasttext, test_x, test_y)
    torch.save(fasttext, os.path.join(corpus_root_path, 'torch_demo', 'fasttext.pkl'))


def run_keras():
    pretreatment = Pretreatment()
    train_x, test_x, train_y, test_y = pretreatment.train_test_split(c=2, test_size=0.1)
    # embedding_matrix = pretreatment.create_embedding_matrix(30000)
    textrnn = FastText(pretreatment.nnparam)
    # textrnn.train(train_x, train_y, embedding_matrix)  # 精度 0.9323043484250149 损失 0.270193725742771
    textrnn.train(train_x, train_y, '')  # 精度 0.9353858005601531 损失 0.2599002837189978
    textrnn.predict(test_x, test_y)


if __name__ == '__main__':
    run_pytorch()
    # run_keras()
