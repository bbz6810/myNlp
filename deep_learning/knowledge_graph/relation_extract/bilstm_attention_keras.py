import os
import numpy as np
import pandas as pd
import keras.backend as K
from keras import layers, models
import matplotlib.pyplot as plt
from keras.utils import to_categorical
from keras.initializers import he_normal
from sklearn.model_selection import train_test_split

from deep_learning.simple_attention.simple_attention_keras import SimpleAttention
from deep_learning.simple_attention.simple_attention2_keras import Attention
from deep_learning.knowledge_graph.relation_extract.load_data import FormatData, word_max_len
from corpus import ner_relation_extract_path
from tools import running_of_time


class BiLSTMAttention:
    def __init__(self, config, embedding_pre):
        f_data = FormatData()
        # f_data.train_test_split()

        config = dict()
        config['embedding_size'] = len(f_data.word2id) + 1
        config['embedding_dim'] = 64
        config['pos_size'] = 82
        config['pos_dim'] = 16
        config['hidden_dim'] = 100
        config['tag_size'] = len(f_data.relation2id)
        config['batch'] = 32
        config['pretrained'] = True

        self.batch = config.get('batch')
        self.embedding_size = config.get('embedding_size')
        self.embedding_dim = config.get('embedding_dim')
        self.hidden_dim = config.get('hidden_dim')
        self.tag_size = config.get('tag_size')
        self.pos_size = config.get('pos_size')
        self.pos_dim = config.get('pos_dim')
        self.pretrained = config.get('pretrained')

    def build_model(self):
        K.clear_session()
        word_input = layers.Input(shape=(word_max_len,))
        pos1_input = layers.Input(shape=(word_max_len,))
        pos2_input = layers.Input(shape=(word_max_len,))
        # relation_input = np.array([i for i in range(self.tag_size)], dtype='float32').repeat(self.batch, 1)
        word_embeds = layers.Embedding(input_dim=self.embedding_size, output_dim=self.embedding_dim)
        pos1_embeds = layers.Embedding(input_dim=self.pos_size, output_dim=self.pos_dim)
        pos2_embeds = layers.Embedding(input_dim=self.pos_size, output_dim=self.pos_dim)
        # relation_embeds = layers.Embedding(input_dim=self.tag_size, output_dim=8)

        words = word_embeds(word_input)
        pos1 = pos1_embeds(pos1_input)
        pos2 = pos2_embeds(pos2_input)

        # relation = relation_embeds(relation_input)

        sample = layers.concatenate([words, pos1, pos2])
        print('input layers shape', sample.shape)
        lstm = layers.Bidirectional(layers.LSTM(units=int(sample.shape[2]), return_sequences=True))(sample)
        lstm = layers.Dropout(0.5)(lstm)
        print('lstm shape', lstm)

        # atten = SimpleAttention()(lstm)
        atten = Attention(step_dim=int(sample.shape[1]))(lstm)
        print('atten', atten.shape)
        atten = layers.Dropout(0.5)(atten)

        output = layers.Dense(self.tag_size, activation='softmax')(atten)
        model = models.Model(input=[word_input, pos1_input, pos2_input], output=output)
        print(model.summary())
        return model

    @running_of_time
    def train(self):
        f_data = FormatData()
        model = self.build_model()
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        for i in range(5):
            print('epoch', i)
            train_datas, test_datas, train_labels, test_labels, train_pos1, test_pos1, train_pos2, test_pos2 = \
                f_data.train_test_split(shuffle=True)
            labels = to_categorical(train_labels)
            test_label = to_categorical(test_labels)
            model.fit(x=[train_datas, train_pos1, train_pos2], y=labels, epochs=8, batch_size=32, verbose=2,
                      validation_data=([test_datas, test_pos1, test_pos2], test_label), shuffle=True)

        model.save(os.path.join(ner_relation_extract_path, 'keras_lstm_att2.model'))

    def predict(self):
        model = models.load_model(os.path.join(ner_relation_extract_path, 'keras_lstm_att2.model'))
        t = FormatData()
        train_datas, test_datas, train_labels, test_labels, train_pos1, test_pos1, train_pos2, test_pos2 = \
            t.train_test_split(train_split=1, shuffle=True)
        labels = to_categorical(train_labels)
        tup = model.evaluate(x=[train_datas, train_pos1, train_pos2], y=labels, batch_size=4096)
        print(tup)


if __name__ == '__main__':
    bl = BiLSTMAttention('', '')
    # bl.build_model()
    # bl.train()
    bl.predict()
