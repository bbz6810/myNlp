import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(1)


class BiLSTMAttention(nn.Module):
    def __init__(self, config, embedding_pre):
        super(BiLSTMAttention, self).__init__()

        self.batch = config.get('batch')
        self.embedding_size = config.get('embedding_size')
        self.embedding_dim = config.get('embedding_dim')
        self.hidden_dim = config.get('hidden_dim')
        self.tag_size = config.get('tag_size')
        self.pos_size = config.get('pos_size')
        self.pos_dim = config.get('pos_dim')

        self.pretrained = config.get('pretrained')
        if self.pretrained:
            self.word_embeds = nn.Embedding.from_pretrained(torch.FloatTensor(embedding_pre), freeze=False)
        else:
            self.word_embeds = nn.Embedding(self.embedding_size, self.embedding_dim)

        self.pos1_embeds = nn.Embedding(self.pos_size, self.pos_dim)
        self.pos2_embeds = nn.Embedding(self.pos_size, self.pos_dim)
        self.relation_embeds = nn.Embedding(self.tag_size, self.hidden_dim)

        self.lstm = nn.LSTM(input_size=self.embedding_dim + self.pos_dim * 2, hidden_size=self.hidden_dim // 2,
                            num_layers=1, bidirectional=True)
        self.hidden2tag = nn.Linear(self.hidden_dim, self.tag_size)

        self.dropout_embeds = nn.Dropout(p=0.5)
        self.dropout_lstm = nn.Dropout(p=0.5)
        self.dropout_att = nn.Dropout(p=0.5)

        self.hidden = self.init_hidden()

        self.attention_weight = nn.Parameter(torch.randn(self.batch, 1, self.hidden_dim))
        self.relation_bias = nn.Parameter(torch.randn(self.batch, self.tag_size, 1))

    def init_hidden(self):
        return torch.randn(2, self.batch, self.hidden_dim // 2)

    def init_hidden_lstm(self):
        return (
            torch.randn(2, self.batch, self.embedding_dim // 2), torch.randn(2, self.batch, self.embedding_dim // 2))

    def attention(self, H):
        M = torch.tanh(H)
        a = F.softmax(torch.bmm(self.attention_weight, M), 2)
        a = torch.transpose(a, 1, 2)
        return torch.bmm(H, a)

    def forward(self, sentence, pos1, pos2):
        self.hidden = self.init_hidden_lstm()
        embeds = torch.cat((self.word_embeds(sentence), self.pos1_embeds(pos1), self.pos2_embeds(pos2)), 2)
        embeds = torch.transpose(embeds, 0, 1)
        lstm_out, self.hidden = self.lstm(embeds, self.hidden)
        lstm_out = torch.transpose(lstm_out, 0, 1)
        lstm_out = torch.transpose(lstm_out, 1, 2)
        lstm_out = self.dropout_lstm(lstm_out)
        att_out = torch.tanh(self.attention(lstm_out))

        relation = torch.tensor([i for i in range(self.tag_size)], dtype=torch.long).repeat(self.batch, 1)
        relation = self.relation_embeds(relation)

        res = torch.add(torch.bmm(relation, att_out), self.relation_bias)
        res = F.softmax(res, 1)
        return res.view(self.batch, -1)
