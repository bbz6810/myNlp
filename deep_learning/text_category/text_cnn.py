import os
from functools import reduce
import torch
import numpy as np
import torch.nn as nn
from torch import optim
from keras import models, layers

from deep_learning.nn_param import NN
from deep_learning.data_pretreatment import Pretreatment
from tools import running_of_time
from corpus import corpus_root_path

batch_size = 32
epochs = 16



class TextCNN2(nn.Module):
    def __init__(self, nn_param):
        super(TextCNN2, self).__init__()
        self.nn_param = nn_param
        self.embedding = nn.Embedding(self.nn_param.vocab_size + 1, self.nn_param.embedding_dim)
        self.embedding.weight.requires_grad = True
        self.convs = nn.ModuleList(
            [nn.Sequential(nn.Conv1d(in_channels=self.nn_param.embedding_dim, out_channels=256, kernel_size=h),
                           nn.ReLU(),
                           nn.MaxPool1d(kernel_size=self.nn_param.max_words - h + 1))
             for h in range(2, 6)])
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(in_features=256 * 4, out_features=self.nn_param.class_num)

    def forward(self, x):
        embed_x = self.embedding(x)
        embed_x = embed_x.permute(0, 2, 1)
        out = [conv(embed_x) for conv in self.convs]
        out = torch.cat(out, dim=1)
        out = out.view(-1, out.size(1))
        out = self.dropout(out)
        out = self.fc(out)
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
    # optimizer = optim.Adam(net.parameters(), lr=lr)
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9)
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
        if idx % 100 == 0:
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


class TextCNN(NN):
    def __init__(self, nn_param):
        super(TextCNN, self).__init__(nn_param=nn_param)

    def model(self, embedding_matrix):
        inputs = layers.Input(shape=(self.nn_param.max_words,), dtype='float32')
        embedding = layers.Embedding(input_dim=self.nn_param.vocab_size + 1, output_dim=self.nn_param.embedding_dim,
                                     input_length=self.nn_param.max_words, trainable=False, weights=[embedding_matrix])
        embed = embedding(inputs)

        cnn1 = layers.Conv1D(256, 3, padding='same', strides=1, activation='relu')(embed)
        cnn1 = layers.MaxPool1D(pool_size=48)(cnn1)

        cnn2 = layers.Conv1D(256, 4, padding='same', strides=1, activation='relu')(embed)
        cnn2 = layers.MaxPool1D(pool_size=47)(cnn2)

        cnn3 = layers.Conv1D(256, 5, padding='same', strides=1, activation='relu')(embed)
        cnn3 = layers.MaxPool1D(pool_size=46)(cnn3)

        cnn = layers.concatenate(inputs=[cnn1, cnn2, cnn3], axis=1)
        flat = layers.Flatten()(cnn)
        drop = layers.Dropout(0.2)(flat)
        if self.nn_param.class_num == 2:
            output = layers.Dense(1, activation='sigmoid')(drop)
        else:
            output = layers.Dense(self.nn_param.class_num, activation='sigmoid')(drop)
        model = models.Model(inputs=inputs, outputs=output)
        print(model.summary())
        return model

    def train(self, train_x, train_y, embedding_matrix):
        self.nn = self.model(embedding_matrix)
        if self.nn_param.class_num == 2:
            self.nn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        else:
            self.nn.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        self.nn.fit(train_x, train_y, epochs=epochs, batch_size=batch_size, validation_split=0.15)

    def predict(self, test_x, test_y):
        _y = self.nn.predict(test_x)
        self.score(_y, test_y)


def run_keras():
    pretreatment = Pretreatment()
    train_x, test_x, train_y, test_y = pretreatment.train_test_split(c=5, y_one_hot=False, test_size=0.6)
    embedding_matrix = pretreatment.create_embedding_matrix(20000)
    textrnn = TextCNN(pretreatment.nnparam)
    textrnn.train(train_x, train_y, embedding_matrix)
    textrnn.predict(test_x, test_y)
    """
    16178/16178 [==============================] - 100s 6ms/step - loss: 0.0672 - acc: 0.9794 - val_loss: 0.2334 - val_acc: 0.9370
    正确个数 26730 总数 28551 正确率 0.9362193968687612
    """


def run_pytorch():
    pretreatment = Pretreatment()
    train_x, test_x, train_y, test_y = pretreatment.train_test_split(c=20, y_one_hot=False, test_size=0.1)
    train_x, train_y, test_x, test_y = data2tensor(train_x, train_y, test_x, test_y)

    textcnn = TextCNN2(pretreatment.nnparam)
    p = 0
    for k, v in textcnn.named_parameters():
        print('name', k, 'param', v.size())
        p += reduce(lambda x, y: x * y, list(v.size()))

    print(textcnn)
    print('总参数量', p)

    for epoch in range(epochs):
        train_model(net=textcnn, train_x=train_x, train_y=train_y, epoch=epoch, lr=0.0001)
        test_model(textcnn, test_x, test_y)
    torch.save(textcnn, os.path.join(corpus_root_path, 'torch_demo', 'textcnn.pkl'))


if __name__ == '__main__':
    run_pytorch()
