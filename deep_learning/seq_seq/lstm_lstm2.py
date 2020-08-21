"""
博客连接:https://blog.csdn.net/weiwei9363/article/details/79464789
keras例子：https://github.com/keras-team/keras/tree/master/examples

将句子转换为3个numpy arrays, encoder_input_data, decoder_input_data, decoder_target_data:
encoder_input_data 是一个 3D 数组，大小为 (num_pairs, max_english_sentence_length, num_english_characters)，包含英语句子的one-hot向量
decoder_input_data 是一个 3D 数组，大小为 (num_pairs, max_fench_sentence_length, num_french_characters) 包含法语句子的one-hot向量
decoder_target_data 与 decoder_input_data 相同，但是有一个时间的偏差。 decoder_target_data[:, t, :] 与decoder_input_data[:, t+1, :]相同
训练一个基于LSTM的Seq2Seq模型，在给定 encoder_input_data和decoder_input_data时，预测 decoder_target_data，我们的模型利用了teacher forcing
解码一些语言用来验证模型事有效的
"""
import json
import numpy as np
from sklearn.model_selection import train_test_split
from keras import layers, models
from keras.callbacks import ModelCheckpoint
import keras.backend as K

from corpus import chinese_to_english_path, french_to_english_path, seq2seq2_model_path, eng2eng_obj_path
from tools.chinese_trans.langconv import cht_to_chs
from corpus.load_corpus import LoadCorpus

batch_size = 32
epochs = 10
latent_dim = 256
samples = 10000

save_params = ['input_char_index', 'output_char_index', 'max_input_seq_length', 'max_output_seq_length',
               'len_input_characters', 'len_output_characters']
save_np_params = ['encoder_input_data', 'decoder_input_data', 'decoder_output_data']


def load_seq2(maxlen):
    x, y = [], []
    with open(french_to_english_path, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            line = line.strip().split('\t')
            x.append(line[0])
            y.append('\t' + line[1] + '\n')
    print('对话语料大小', len(x))
    return x[:maxlen], y[:maxlen]


class Eng2Eng:
    def __init__(self, samples=10000):
        self.samples = samples

        self.encoder_input_data = None
        self.decoder_input_data = None
        self.decoder_output_data = None

        self.max_input_seq_length = 0
        self.max_output_seq_length = 0

        self.len_input_characters = 0
        self.len_output_characters = 0

        self.input_char_index = None
        self.output_char_index = None

        self.encoder_model = None
        self.decoder_model = None

    def prepare_data(self):
        input_characters = set()
        target_characters = set()

        x, y = load_seq2(self.samples)
        for _x, _y in zip(x, y):
            for cx in _x:
                input_characters.add(cx)
            for cy in _y:
                target_characters.add(cy)

        self.max_input_seq_length = max(len(t) for t in x)
        self.max_output_seq_length = max(len(t) for t in y)

        self.len_input_characters = len(input_characters)
        self.len_output_characters = len(target_characters)

        print('输入字符集长度', self.len_input_characters)
        print('输出字符集长度', self.len_output_characters)
        print('输入字符串最大长度', self.max_input_seq_length)
        print('输出字符串最大长度', self.max_output_seq_length)

        self.input_char_index = {char: i for i, char in enumerate(sorted(list(input_characters)))}
        self.output_char_index = {char: i for i, char in enumerate(sorted(list(target_characters)))}

        self.encoder_input_data = np.zeros(shape=(len(x), self.max_input_seq_length, self.len_input_characters),
                                           dtype='float32')
        self.decoder_input_data = np.zeros(shape=(len(x), self.max_output_seq_length, self.len_output_characters),
                                           dtype='float32')
        self.decoder_output_data = np.zeros(shape=(len(x), self.max_output_seq_length, self.len_output_characters),
                                            dtype='float32')

        print('编码后输入数据形状.shape', self.encoder_input_data.shape)
        print('解码后输入数据形状.shape', self.decoder_input_data.shape)
        print('解码后输出数据形状.shape', self.decoder_output_data.shape)

        for i, (input_text, target_text) in enumerate(zip(x, y)):
            for t, char in enumerate(input_text):
                self.encoder_input_data[i, t, self.input_char_index[char]] = 1.0
            for t, char in enumerate(target_text):
                self.decoder_input_data[i, t, self.output_char_index[char]] = 1.0
                if t > 0:
                    self.decoder_output_data[i, t - 1, self.output_char_index[char]] = 1.0

    def build_model(self):
        # 编码器 = 输入层 + LSTM层
        encoder_inputs = layers.Input(shape=(None, self.len_input_characters), name='encoder_inputs')
        encoder_lstm = layers.LSTM(latent_dim, return_state=True, name='encoder_lstm')
        encoder_outputs, state_h, state_c = encoder_lstm(encoder_inputs)
        encoder_state = [state_h, state_c]

        # 解码器 = 输出层 + LSTM层 + Dense层
        decoder_inputs = layers.Input(shape=(None, self.len_output_characters), name='decoder_inputs')
        decoder_lstm = layers.LSTM(latent_dim, return_state=True, return_sequences=True, name='decoder_lstm')
        decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_state)
        decoder_dense = layers.Dense(self.len_output_characters, activation='softmax', name='decoder_dense')
        decoder_outputs = decoder_dense(decoder_outputs)

        model = models.Model(inputs=[encoder_inputs, decoder_inputs], outputs=decoder_outputs)
        model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
        print(model.summary())
        return model

    def train(self):
        self.prepare_data()
        model = self.build_model()
        checkpoint = ModelCheckpoint(filepath=seq2seq2_model_path, save_best_only=True)
        model.fit(x=[self.encoder_input_data, self.decoder_input_data], y=self.decoder_output_data,
                  batch_size=batch_size, epochs=epochs, validation_split=0.2, callbacks=[checkpoint])
        self.save(model, seq2seq2_model_path, eng2eng_obj_path)

    def load_model(self, seq2seq_model_path):
        model = models.load_model(seq2seq_model_path)
        encoder_inputs = model.get_layer('encoder_inputs').input
        encoder_lstm = model.get_layer('encoder_lstm')
        encoder_outputs, i_state_h, i_state_c = encoder_lstm(encoder_inputs)
        encoder_state = [i_state_h, i_state_c]
        self.encoder_model = models.Model(encoder_inputs, encoder_state)

        decoder_inputs = model.get_layer('decoder_inputs').input
        decoder_lstm = model.get_layer('decoder_lstm')
        decoder_dense = model.get_layer('decoder_dense')

        decoder_state_input_h = layers.Input(shape=(latent_dim,))
        decoder_state_input_c = layers.Input(shape=(latent_dim,))
        decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

        decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)
        decoder_states = [state_h, state_c]
        decoder_outputs = decoder_dense(decoder_outputs)
        self.decoder_model = models.Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)

    def save(self, model, model_path, obj_save_path):
        model.save(model_path)

        d = {}
        for param in save_params:
            print(param)
            d[param] = getattr(self, param)
        json.dump(d, open(obj_save_path, mode='w'))

        for param in save_np_params:
            np.save('{}_{}'.format(obj_save_path, param), getattr(self, param))

    def load(self, model_path, obj_save_path):
        self.load_model(model_path)
        d = json.load(open(obj_save_path, mode='r'))
        for param in save_params:
            setattr(self, param, d[param])
        for param in save_np_params:
            setattr(self, param, np.load('{}_{}.npy'.format(obj_save_path, param)))

    def predict(self):
        def input_sentence_vector(self, sentence):
            input_vector = np.zeros(shape=(1, self.max_input_seq_length, self.len_input_characters))
            for index, char in enumerate(sentence):
                input_vector[0, index, self.input_char_index[char]] = 1.0
            decoded_sentence = self.decode_sequence(input_vector, reverse_target_char_index)
            return decoded_sentence

        self.load(seq2seq2_model_path, eng2eng_obj_path)
        reverse_target_char_index = {i: char for char, i in self.output_char_index.items()}
        while True:
            sentence = input('question:')
            retv = input_sentence_vector(self, sentence)
            print('answer:', retv)

    def decode_sequence(self, input_seq, reverse_target_char_index):
        states_value = self.encoder_model.predict(input_seq)
        # print('states value', states_value)
        target_seq = np.zeros(shape=(1, 1, self.len_output_characters))
        target_seq[0, 0, self.output_char_index['\t']] = 1.

        stop = False
        decoded_sentence = ''

        while not stop:
            output_tokens, h, c = self.decoder_model.predict([target_seq] + states_value)

            sampled_token_index = np.argmax(output_tokens[0, -1, :])
            sampled_char = reverse_target_char_index[sampled_token_index]
            decoded_sentence += sampled_char

            if sampled_char == '\n' or len(decoded_sentence) > self.max_output_seq_length:
                stop = True

            target_seq = np.zeros(shape=(1, 1, self.len_output_characters))
            target_seq[0, 0, sampled_token_index] = 1.

            states_value = [h, c]
        return decoded_sentence


if __name__ == '__main__':
    eng = Eng2Eng()
    # eng.train()
    eng.predict()
