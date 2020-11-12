import math
import torch
import numpy as np
import random
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch import optim
from torch.nn.utils import clip_grad_norm_

from corpus.load_corpus import LoadCorpus
from deep_learning.pytorch_net_base_cls import numpy2tensor, data_iter


class Encoder(nn.Module):
    def __init__(self, input_size, embed_size, hidden_size, n_layer=1, dropout=0.5):
        super(Encoder, self).__init__()
        self.input_size = input_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size

        self.embed = nn.Embedding(self.input_size, self.embed_size)
        self.gru = nn.GRU(self.embed_size, self.hidden_size, num_layers=n_layer, dropout=dropout, bidirectional=True)

    def forward(self, x, hidden=None):
        embedded = self.embed(x)
        outputs, hidden = self.gru(embedded, hidden)
        # sum bidirectional outputs
        outputs = (outputs[:, :, :self.hidden_size] + outputs[:, :, :self.hidden_size])
        return outputs, hidden


class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size

        self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
        self.v = nn.Parameter(torch.rand(self.hidden_size))
        stdv = 1. / math.sqrt(self.v.size(0))
        self.v.data.uniform_(-stdv, stdv)

    def forward(self, x, hidden):
        timestep = x.size(0)
        h = hidden.repeat(timestep, 1, 1).transpose(0, 1)
        x = x.transpose(0, 1)
        attn_energies = self.score(x, h)
        return F.softmax(attn_energies, dim=1).unsqueeze(1)

    def score(self, x, hidden):
        energy = F.relu(self.attn(torch.cat([hidden, x], 2)))
        energy = energy.transpose(1, 2)
        v = self.v.repeat(x.size(0), 1).unsqueeze(1)
        energy = torch.bmm(v, energy)
        return energy.squeeze(1)


class Decoder(nn.Module):
    def __init__(self, embed_size, hidden_size, output_size, n_layer=1, dropout=0.2):
        super(Decoder, self).__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layer = n_layer

        self.embed = nn.Embedding(self.output_size, self.embed_size)
        self.dropout = nn.Dropout(dropout, inplace=True)
        self.attention = Attention(self.hidden_size)
        self.gru = nn.GRU(self.hidden_size + self.embed_size, self.hidden_size, n_layer, dropout=dropout)
        self.fc = nn.Linear(self.hidden_size * 2, self.output_size)

    def forward(self, x, last_hidden, encoder_outputs):
        embedded = self.embed(x).unsqueeze(0)
        embedded = self.dropout(embedded)
        attn_weights = self.attention(encoder_outputs, last_hidden[-1])
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))
        context = context.transpose(0, 1)

        rnn_input = torch.cat([embedded, context], 2)
        output, hidden = self.gru(rnn_input, last_hidden)
        output = output.squeeze(0)
        context = context.squeeze(0)
        output = self.fc(torch.cat([output, context], 1))
        output = F.log_softmax(output, dim=1)
        return output, hidden, attn_weights


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        batch_size = src.size(1)
        max_len = trg.size(0)
        vocab_size = self.decoder.output_size
        # todo
        outputs = Variable(torch.zeros(max_len, batch_size, vocab_size))
        encoder_output, hidden = self.encoder(src)
        # todo: others
        hidden = hidden[:self.decoder.n_layer]
        output = Variable(trg.data[0, :])  # sos
        for t in range(1, max_len):
            output, hidden, attn_weights = self.decoder(output, hidden, encoder_output)
            outputs[t] = output
            is_teacher = random.random() < teacher_forcing_ratio
            top1 = output.data.max(1)[1]
            output = Variable(trg.data[t] if is_teacher else top1)
        return outputs


def load_data():
    word2id = {'sos': 0, 'eos': 1}
    x, y = LoadCorpus.load_chatbot100_train()
    max_len = 20
    for i, j in zip(x, y):
        for _i in i.split():
            if _i not in word2id:
                word2id[_i] = len(word2id)
        for _j in j.split():
            if _j not in word2id:
                word2id[_j] = len(word2id)
    id2word = dict((v, k) for k, v in word2id.items())
    vocab_size = len(word2id)

    train_x, train_y = [], []
    for i, j in zip(x, y):
        t_x = [word2id[_i] for _i in i.split()]
        t_y = [word2id[_j] for _j in j.split()]
        if len(t_x) > max_len:
            t_x = [0] + t_x[:max_len - 2] + [1]
        else:
            t_x = [0] + t_x + [1] * (max_len - len(t_x) - 1)

        if len(t_y) > max_len:
            t_y = t_y[:max_len]
        else:
            t_y = t_y + [0] * (max_len - len(t_y))
        train_x.append(t_x)
        train_y.append(t_y)
    train_x = numpy2tensor(np.array(train_x))
    train_y = numpy2tensor(np.array(train_y))
    return train_x, train_y, vocab_size


def train(model, vocab_size, train_x, train_y):
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    model.train()
    pad = 0
    for i in range(1000):
        for idx, (x, y) in enumerate(data_iter(train_x, train_y, batch_size=4)):
            optimizer.zero_grad()
            x, y = x.transpose(0, 1), y.transpose(0, 1)
            outputs = model(x, y)
            loss = F.nll_loss(outputs[1:].view(-1, vocab_size), y[1:].contiguous().view(-1), ignore_index=pad)
            loss.backward()
            clip_grad_norm_(model.parameters(), 10)
            optimizer.step()
            loss += loss.data.item()
            print('loss', loss.item())


if __name__ == '__main__':
    train_x, train_y, vocab_size = load_data()
    encoder = Encoder(input_size=vocab_size, embed_size=64, hidden_size=64, n_layer=2, dropout=0.5)
    decoder = Decoder(embed_size=64, hidden_size=64, output_size=vocab_size, n_layer=1, dropout=0.5)
    seq2seq = Seq2Seq(encoder=encoder, decoder=decoder)
    train(seq2seq, vocab_size, train_x, train_y)
