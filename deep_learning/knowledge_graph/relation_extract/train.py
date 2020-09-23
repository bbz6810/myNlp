import json
import os
import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as D
from torch.autograd import Variable

from corpus import chinese_ner_path
from deep_learning.knowledge_graph.relation_extract.bilstm_attention import BiLSTMAttention
from deep_learning.knowledge_graph.relation_extract.load_data import FormatData
from tools import running_of_time

epochs = 100


@running_of_time
def running():
    f_data = FormatData()
    train_datas, test_datas, train_labels, test_labels, train_pos1, test_pos1, train_pos2, test_pos2 = f_data.train_test_split()

    config = dict()
    config['embedding_size'] = len(f_data.word2id) + 1
    config['embedding_dim'] = 100
    config['pos_size'] = 82
    config['pos_dim'] = 25
    config['hidden_dim'] = 100
    config['tag_size'] = len(f_data.relation2id)
    config['batch'] = 32
    config['pretrained'] = True

    print('nn config', config)

    # embedding_pre = list()
    # if config['pretrained'] is True:
    #     print('use pretrained embedding')
    #     word2vec = dict()
    #     with open(os.path.join(chinese_ner_path, 'vec.txt'), mode='r', encoding='utf8') as f:
    #         for line in f.readlines():
    #             line = line.strip().split()
    #             word2vec[line[0]] = line[1]
    #
    #     unknow_pre = np.ones(shape=(1, 100), dtype='float32')
    #     embedding_pre = np.zeros(shape=(len(word2vec), 100), dtype='float32')
    #     embedding_pre[0, :] = unknow_pre
    #     idx = 1
    #     for word in f_data.word2id:
    #         if word2vec.get(word):
    #             embedding_pre[idx, :] = np.array(word2vec[word])
    #         else:
    #             embedding_pre[idx, :] = unknow_pre
    #         idx += 1
    #     embedding_pre = np.array(embedding_pre[:idx], dtype='float32')
    #     print('pretrain embedding shape', embedding_pre.shape)
    #
    # learning_rate = 0.0005
    # model = BiLSTMAttention(config, embedding_pre=embedding_pre)
    # optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    # criterion = nn.CrossEntropyLoss(reduction='mean')
    #
    # train = torch.LongTensor(train_datas[:len(train_datas) - len(train_datas) % config['batch']])
    # position1 = torch.LongTensor(train_pos1[:len(train_pos1) - len(train_pos1) % config['batch']])
    # position2 = torch.LongTensor(train_pos2[:len(train_pos2) - len(train_pos2) % config['batch']])
    # labels = torch.LongTensor(train_labels[:len(train_labels) - len(train_labels) % config['batch']])
    # train_datasets = D.TensorDataset(train, position1, position2, labels.reshape(-1, ))
    # train_dataloader = D.DataLoader(dataset=train_datasets, batch_size=config['batch'], shuffle=True, num_workers=4)
    #
    # for epoch in range(epochs):
    #     print('epoch', epoch)
    #     acc, total = 0, 0
    #
    #     for sentence, pos1, pos2, lab in train_dataloader:
    #         print(acc, total)
    #         sentence = Variable(sentence)
    #         pos1 = Variable(pos1)
    #         pos2 = Variable(pos2)
    #         lab = Variable(lab)
    #         y = model(sentence, pos1, pos2)
    #         loss = criterion(y, lab)
    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()
    #
    #         y = np.argmax(y.data.numpy(), axis=1)
    #
    #         for y1, y2 in zip(y, lab):
    #             if y1 == y2:
    #                 acc += 1
    #             total += 1
    #     print('train', acc / total * 100)
    # torch.save(model, os.path.join(chinese_ner_path, 'torch_att_{:.2f}'.format(acc / total * 100)))

    model = torch.load(os.path.join(chinese_ner_path, 'torch_att_87.51'))

    test = torch.LongTensor(test_datas[:len(test_datas) - len(test_datas) % config['batch']])
    pos1 = torch.LongTensor(test_pos1[:len(test_pos1) - len(test_pos1) % config['batch']])
    pos2 = torch.LongTensor(test_pos2[:len(test_pos2) - len(test_pos2) % config['batch']])
    labs = torch.LongTensor(test_labels[:len(test_labels) - len(test_labels) % config['batch']])
    print('test', test.shape)
    print('pos1', pos1.shape)
    print('pos2', pos2.shape)
    print('labs', labs.shape)
    test_datasets = D.TensorDataset(test, pos1, pos2, labs.reshape(-1, ))
    test_dataloader = D.DataLoader(dataset=test_datasets, batch_size=config['batch'], shuffle=True, num_workers=4)

    acc_t, total_t = 0, 0
    count_predict = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    count_total = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    count_right = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    predict, total, right = 0, 0, 0

    for sentence, p1, p2, lab in test_dataloader:
        sentence = Variable(sentence)
        p1 = Variable(p1)
        p2 = Variable(p2)
        lab = Variable(lab)
        y = model(sentence, p1, p2)
        y = np.argmax(y.data.numpy(), axis=1)
        for y1, y2 in zip(y, lab):
            count_predict[y1] += 1
            count_total[y2] += 1
            predict += 1
            total += 1
            if y1 == y2:
                count_right[y1] += 1
                right += 1

    precision = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    recall = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    for i in range(len(count_predict)):
        if count_predict[i] != 0:
            precision[i] = count_right[i] / count_predict[i]

        if count_total[i] != 0:
            recall[i] = count_right[i] / count_total[i]

    p = right / predict
    r = right / total

    precition = sum(precision) / len(f_data.relation2id)
    recall = sum(recall) / len(f_data.relation2id)
    print('准确率', precision)
    print('召回率', recall)
    # print('F1', (2 * precision * recall / (precision + recall)))
    print('F1', [(2 * i * recall) / (i + recall) for i in precision])

    print('p', p, 'r', r, 'f1', (2 * p * r) / (p + r))


if __name__ == '__main__':
    # running()
    from deep_learning.knowledge_graph.relation_extract.bilstm_attention_keras import BiLSTMAttention

    b = BiLSTMAttention('', '')
    b.build_model()
