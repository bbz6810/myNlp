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
from deep_learning.knowledge_graph.relation_extract.load_data import FormatData, word_max_len
from corpus import ner_relation_extract_path
from tools import running_of_time

SINGLE_ATTENTION_VECTOR = True
APPLY_ATTENTION_BEFORE_LSTM = False
INPUT_DIM = 2
TIME_STEPS = 20


def get_data_recurrent(n, time_steps, input_dim, attention_column=10):
    x = np.random.standard_normal(size=(n, time_steps, input_dim))
    y = np.random.randint(low=0, high=2, size=(n, 1))
    x[:, attention_column, :] = np.tile(y[:], (1, input_dim))
    return x, y


def get_activations(model, inputs, print_shape_only=False, layer_name=None):
    activations = []
    inp = model.input
    if layer_name is None:
        outputs = [layer.output for layer in model.layers]
    else:
        outputs = [layer.output for layer in model.layers if layer.name == layer_name]
    funcs = [K.function([inp] + [K.learning_phase()], [out]) for out in outputs]
    layer_outputs = [func([inputs, 1])[0] for func in funcs]
    for layer_activations in layer_outputs:
        activations.append(layer_activations)
    return activations


def attention_3d_block(inputs):
    # inputs.shape = (batch_size, time_steps, input_dim)
    print('input shape', inputs.shape)
    input_dim = int(inputs.shape[2])
    a = layers.Permute((2, 1))(inputs)
    a = layers.Reshape((input_dim, TIME_STEPS))(
        a)  # this line is not useful. It's just to know which dimension is what.
    a = layers.Dense(TIME_STEPS, activation='softmax')(a)
    if SINGLE_ATTENTION_VECTOR:
        a = layers.Lambda(lambda x: K.mean(x, axis=1), name='dim_reduction')(a)
        a = layers.RepeatVector(input_dim)(a)
    a_probs = layers.Permute((2, 1), name='attention_vec')(a)
    output_attention_mul = layers.Multiply()([inputs, a_probs])
    return output_attention_mul


def model_attention_applied_after_lstm():
    K.clear_session()  # 清除之前的模型，省得压满内存
    inputs = layers.Input(shape=(TIME_STEPS, INPUT_DIM,))
    lstm_units = 32
    lstm_out = layers.LSTM(lstm_units, return_sequences=True)(inputs)
    print('lstm out shape', lstm_out.shape)
    attention_mul = attention_3d_block(lstm_out)
    attention_mul = layers.Flatten()(attention_mul)
    output = layers.Dense(1, activation='sigmoid')(attention_mul)
    model = models.Model(input=[inputs], output=output)
    return model


def model_attention_applied_before_lstm():
    K.clear_session()  # 清除之前的模型，省得压满内存
    inputs = layers.Input(shape=(TIME_STEPS, INPUT_DIM,))
    attention_mul = attention_3d_block(inputs)
    lstm_units = 32
    attention_mul = layers.LSTM(lstm_units, return_sequences=False)(attention_mul)
    output = layers.Dense(1, activation='sigmoid')(attention_mul)
    model = models.Model(input=[inputs], output=output)
    return model


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
        print('输入映射层 shape', sample.shape)

        sample_dim3 = int(sample.shape[2])
        lstm = layers.Bidirectional(layers.LSTM(units=sample_dim3, return_sequences=True))(sample)
        print('lstm shape', lstm)

        # # attention
        # print('attention lstm shape', lstm.shape)
        # input_dim = int(lstm.shape[2])
        # lstm = layers.Permute((2, 1))(lstm)
        # lstm = layers.Dense(sample_dim3 * 2, activation='softmax')(lstm)
        # if SINGLE_ATTENTION_VECTOR:
        #     lstm = layers.Lambda(lambda x: K.mean(x, axis=1), name='dim_reduction')(lstm)
        #     lstm = layers.RepeatVector(input_dim)(lstm)
        # lstm_probs = layers.Permute((2, 1), name='attention_vec')(lstm)
        # atten = layers.Multiply()([lstm, lstm_probs])
        # print('atten', atten.shape)

        atten = SimpleAttention()(lstm)
        print('atten', atten.shape)

        # flat = layers.Flatten()(atten)

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
    # BiLSTMAttention('', '').build_model()
    # BiLSTMAttention('', '').train()
    BiLSTMAttention('', '').predict()

    # np.random.seed(1337)  # for reproducibility
    #
    # # if True, the attention vector is shared across the input_dimensions where the attention is applied.
    #
    # N = 300000
    # # N = 300 -> too few = no training
    # inputs_1, outputs = get_data_recurrent(N, TIME_STEPS, INPUT_DIM)
    # if APPLY_ATTENTION_BEFORE_LSTM:
    #     m = model_attention_applied_before_lstm()
    # else:
    #     m = model_attention_applied_after_lstm()
    #
    # m.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    # m.summary()
    #
    # m.fit([inputs_1], outputs, epochs=1, batch_size=64, validation_split=0.1)
    #
    # m.save(os.path.join(ner_relation_extract_path, 'lstm_attention.model'))

    # m = models.load_model(os.path.join(ner_relation_extract_path, 'lstm_attention.model'))

    # attention_vectors = []
    # for i in range(300):
    #     testing_inputs_1, testing_outputs = get_data_recurrent(1, TIME_STEPS, INPUT_DIM)
    #     attention_vector = np.mean(get_activations(m,
    #                                                testing_inputs_1,
    #                                                print_shape_only=True,
    #                                                layer_name='attention_vec')[0], axis=2).squeeze()
    #     #        print('attention =', attention_vector)
    #     assert (np.sum(attention_vector) - 1.0) < 1e-5
    #     attention_vectors.append(attention_vector)
    #
    # attention_vector_final = np.mean(np.array(attention_vectors), axis=0)
    # # plot part.
    # print(attention_vector_final)

    # pd.DataFrame(attention_vector_final, columns=['attention (%)']).plot(kind='bar',
    #                                                                      title='Attention Mechanism as '
    #                                                                            'a function of input'
    #                                                                            ' dimensions.')
    # plt.show()
