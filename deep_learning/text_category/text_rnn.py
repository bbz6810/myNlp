import os
import sys
import torch.nn as nn
import torch
from torch import optim
import numpy as np
from keras import models, layers
from deep_learning.data_pretreatment import Pretreatment
from deep_learning.nn_param import NN
from tools import running_of_time
from corpus import corpus_root_path

sys.path.append('/Users/zhoubb/projects/myNlp')

batch_size = 32
epochs = 4
lstm_dim = 128


class TextRNN(NN):
    def __init__(self, nn_param):
        super(TextRNN, self).__init__(nn_param=nn_param)

    def model(self, embedding_matrix):
        inputs = layers.Input(shape=(self.nn_param.max_words,))
        wv = layers.Embedding(self.nn_param.vocab_size + 1, self.nn_param.embedding_dim, weights=[embedding_matrix])(
            inputs)
        lstm = layers.LSTM(lstm_dim)(wv)
        if self.nn_param.class_num == 2:
            y = layers.Dense(1, activation='sigmoid')(lstm)
        else:
            y = layers.Dense(self.nn_param.class_num, activation='softmax')(lstm)
        model = models.Model(input=inputs, output=y)
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


class TextRNN2(nn.Module):
    def __init__(self, nn_param):
        super(TextRNN2, self).__init__()
        self.nn_param = nn_param
        self.hidden_size = 256
        self.layer_num = 1
        self.bidirectional = True
        self.embedding = nn.Embedding(self.nn_param.vocab_size + 1, self.nn_param.embedding_dim)
        self.embedding.weight.requires_grad = True
        self.lstm = nn.LSTM(self.nn_param.embedding_dim, self.hidden_size, self.layer_num, batch_first=True,
                            bidirectional=self.bidirectional)
        self.fc = nn.Linear(self.hidden_size * 2, self.nn_param.class_num) if self.bidirectional else nn.Linear(
            self.hidden_size, self.nn_param.class_num)

    def forward(self, x):
        x = self.embedding(x)
        h0 = torch.zeros(self.layer_num * 2, x.size(0), self.hidden_size) if self.bidirectional else torch.zeros(
            self.layer_num, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.layer_num * 2, x.size(0), self.hidden_size) if self.bidirectional else torch.zeros(
            self.layer_num, x.size(0), self.hidden_size)
        out, (hn, cn) = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out


class SelfAttention(nn.Module):
    def __init__(self, hidden_size):
        super(SelfAttention, self).__init__()
        self.hidden_size = hidden_size
        self.class_num = 1
        self.projection = nn.Sequential(
            nn.Linear(self.hidden_size, 64),
            nn.ReLU(True),
            nn.Linear(64, self.class_num)
        )

    def forward(self, x):
        energy = self.projection(x)
        weights = nn.functional.softmax(energy.squeeze(-1), dim=1)
        outputs = (x * weights.unsqueeze(-1)).sum(dim=1)
        return outputs, weights


class TextRNN3(nn.Module):
    def __init__(self, nn_param):
        super(TextRNN3, self).__init__()
        self.nn_param = nn_param
        self.input_dim = nn_param.vocab_size + 1
        self.embedding_dim = nn_param.embedding_dim
        self.hidden_size = 256
        self.embedding = nn.Embedding(self.input_dim, self.embedding_dim)
        self.lstm = nn.LSTM(self.embedding_dim, self.hidden_size, bidirectional=True)
        self.attention = SelfAttention(self.hidden_size)
        self.fc = nn.Linear(self.hidden_size, 1)

    def forward(self, x):
        embedded = self.embedding(x)
        current_batch_size = x.size(0)
        packed_emb = nn.utils.rnn.pack_sequence(embedded, current_batch_size)
        out, hidden = self.lstm(packed_emb)
        out = nn.utils.rnn.pad_packed_sequence(out)[0]
        out = out[:, :, :self.hidden_size] + out[:, :, self.hidden_size:]
        embedding, attn_weights = self.attention(out.transpose(0, 1))
        outputs = self.fc(embedding.view(current_batch_size, -1))
        return outputs, attn_weights


class BiGRUAttention(nn.Module):
    def __init__(self, nn_param):
        super(BiGRUAttention, self).__init__()
        self.hidden_size = 256
        self.gru_layer = 2
        self.nn_param = nn_param

        self.embedding = nn.Embedding(nn_param.vocab_size + 1, nn_param.embedding_dim)
        self.bigru = nn.GRU(input_size=self.nn_param.embedding_dim, hidden_size=self.hidden_size // 2,
                            num_layers=self.gru_layer, bidirectional=True, batch_first=True)
        self.weight_w = nn.Parameter(torch.Tensor(self.hidden_size, self.hidden_size))
        self.weight_proj = nn.Parameter(torch.Tensor(self.hidden_size, 1))

        self.fc = nn.Linear(self.hidden_size, self.nn_param.class_num)

        nn.init.uniform_(self.weight_w, -0.1, 0.1)
        nn.init.uniform_(self.weight_proj, -0.1, 0.1)

    def forward(self, x):
        embeds = self.embedding(x)
        gru_output, hidden = self.bigru(embeds)
        x = gru_output
        # x = gru_output.permute(1, 0, 2)

        u = torch.tanh(torch.matmul(x, self.weight_w))
        att = torch.matmul(u, self.weight_proj)
        att_socre = nn.functional.softmax(att, dim=1)
        score_x = x * att_socre

        feat = torch.sum(score_x, dim=1)
        y = self.fc(feat)
        return y


def data_iter(train_x, train_y):
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
def train_model(net, train_x, train_y, epoch, lr):
    print('begin training epoch', epoch)
    l_train = len(train_x)
    net.train()
    optimizer = optim.Adam(net.parameters(), lr=lr)
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


@running_of_time
def train_model2(net, train_x, train_y, epoch, lr):
    print('begin training epoch', epoch)
    l_train = len(train_x)
    net.train()
    optimizer = optim.Adam(net.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    # criterion = nn.BCEWithLogitsLoss()

    num_correct = 0
    for idx, (x, y) in enumerate(data_iter(train_x, train_y)):
        optimizer.zero_grad()
        output, _ = net(x)
        y = y.view(-1, 1)
        loss = criterion(output, y.float())
        loss.backward()
        optimizer.step()

        pred = torch.round(torch.sigmoid(output))
        num_correct += (pred == y).float().sum().item()
        if idx % 10 == 0:
            print('train epoch={}, samples={}->[{:.2%}], loss={:.6}'.format(epoch, idx * batch_size,
                                                                            idx * batch_size / l_train,
                                                                            loss.item() / batch_size))
    print('train epoch={}, correct={:.2%}'.format(epoch, num_correct / l_train))


def test_model2(net, test_x, test_y):
    print('begin rnn test', len(test_x))
    net.eval()
    correct = 0
    with torch.no_grad():
        for idx, (x, y) in enumerate(data_iter(test_x, test_y)):
            output, _ = net(x)
            pred = torch.round(torch.sigmoid(output))
            correct += (pred.view(-1, ) == y).float().sum().item()
            print(correct)
            # return
    print('test accuracy={:.2%}'.format(correct / len(test_x)))


def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print({'Total': total_num, 'Trainable': trainable_num})


def run_keras():
    pretreatment = Pretreatment()
    train_x, test_x, train_y, test_y = pretreatment.train_test_split(c=3, test_size=0.6)
    embedding_matrix = pretreatment.create_embedding_matrix(15000)
    textrnn = TextRNN(pretreatment.nnparam)
    textrnn.train(train_x, train_y, embedding_matrix)
    textrnn.predict(test_x, test_y)


def run_pytorch():
    pretreatment = Pretreatment()
    train_x, test_x, train_y, test_y = pretreatment.train_test_split(c=3, y_one_hot=False, test_size=0.1)
    train_x, train_y, test_x, test_y = data2tensor(train_x, train_y, test_x, test_y)

    m = TextRNN2(pretreatment.nnparam)
    # m = BiGRUAttention(pretreatment.nnparam)
    print(m)
    get_parameter_number(m)

    for epoch in range(epochs):
        train_model(net=m, train_x=train_x, train_y=train_y, epoch=epoch, lr=0.0001)
        test_model(m, test_x, test_y)
    torch.save(m, os.path.join(corpus_root_path, 'torch_demo', 'textrnn_attention.pkl'))


if __name__ == '__main__':
    run_pytorch()
    # pretreatment = Pretreatment()
    # train_x, test_x, train_y, test_y = pretreatment.train_test_split(c=2, test_size=0.1, random_state=12)
    # train_x, train_y, test_x, test_y = data2tensor(train_x, train_y, test_x, test_y)
    # m = torch.load(os.path.join(corpus_root_path, 'torch_demo', 'textrnn_attention.pkl'))
    # test_model(m, test_x, test_y)
