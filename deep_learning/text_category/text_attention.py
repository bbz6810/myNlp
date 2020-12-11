import os
import sys
import torch.nn as nn
import torch
from deep_learning.data_pretreatment import Pretreatment
from corpus import corpus_root_path
from deep_learning.pytorch_net_base_cls import data2tensor, get_parameter_number, TrainTorch

sys.path.append('/Users/zhoubb/projects/myNlp')

batch_size = 32
epochs = 16


class BiGRUAttention(nn.Module):
    def __init__(self, nn_param):
        super(BiGRUAttention, self).__init__()
        self.hidden_size = 256
        self.gru_layer = 2
        self.nn_param = nn_param
        self.instance_name = 'BiGRUAttention'

        self.embedding = nn.Embedding(nn_param.vocab_size + 1, nn_param.embedding_dim)
        self.bigru = nn.GRU(input_size=self.nn_param.embedding_dim, hidden_size=self.hidden_size // 2,
                            num_layers=self.gru_layer, bidirectional=True)
        self.weight_w = nn.Parameter(torch.Tensor(self.hidden_size, self.hidden_size))
        self.weight_proj = nn.Parameter(torch.Tensor(self.hidden_size, 1))

        self.fc = nn.Linear(self.hidden_size, self.nn_param.class_num)

        nn.init.uniform_(self.weight_w, -0.1, 0.1)
        nn.init.uniform_(self.weight_proj, -0.1, 0.1)

    def forward(self, x):
        print('x0', x.size())
        embeds = self.embedding(x)
        x, hidden = self.bigru(embeds)
        print('x', x.size())

        o = (x[:, :, :self.hidden_size // 2] + x[:, :, self.hidden_size // 2:])
        print('o', o.size())

        # x = gru_output.permute(1, 0, 2)

        u = torch.tanh(torch.matmul(x, self.weight_w))
        print('u', u.size())
        att = torch.matmul(u, self.weight_proj)
        print('att', att.size())
        att_socre = nn.functional.softmax(att, dim=1)
        print('att_score', att_socre.size())
        score_x = x * att_socre

        feat = torch.sum(score_x, dim=1)
        y = self.fc(feat)
        return y


def run_pytorch():
    pretreatment = Pretreatment()
    train_x, test_x, train_y, test_y = pretreatment.train_test_split(c=2, test_size=0.1)
    train_x, train_y, test_x, test_y = data2tensor(train_x, train_y, test_x, test_y)

    m = BiGRUAttention(pretreatment.nnparam)
    print(m)
    get_parameter_number(m)

    train_model = TrainTorch(model=m)
    train_model.train(train_set=(train_x[:1000], train_y[:1000]), test_set=(test_x, test_y), batch_size=batch_size,
                      epochs=epochs,
                      lr=0.0001, use_test=True)

    torch.save(m, os.path.join(corpus_root_path, 'torch_demo', 'text_{}.pkl'.format(m.instance_name)))


if __name__ == '__main__':
    run_pytorch()
