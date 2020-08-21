import json
import jieba
import numpy as np
from sklearn.model_selection import train_test_split
from keras import layers, models
from keras.initializers import he_normal
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

from corpus import chinese_to_english_path, french_to_english_path, seq2seq2_model_path, word2word_obj_path
from tools.chinese_trans.langconv import cht_to_chs
from corpus.load_corpus import LoadCorpus

batch_size = 32
epochs = 10
embedding_dim = 64
latent_dim = 256
samples = 1000000
input_max_len = 20
output_max_len = 20

start = '\t'
end = '\n'

save_params = ['input_char_index', 'output_char_index', 'max_input_seq_length', 'max_output_seq_length',
               'len_input_characters', 'len_output_characters']
save_np_params = ['encoder_input_data', 'decoder_input_data', 'decoder_output_data']


def load_cha_cha(maxlen):
    c = 0
    x, y = [], []
    for _x, _y in zip(*LoadCorpus.load_xiaohuangji_train()):
        if len(''.join(_x)) > input_max_len or len(''.join(_y)) > input_max_len:
            continue
        # x.append(['START'] + jieba.lcut(''.join(_x.split())) + ['END'])
        # y.append(['START'] + jieba.lcut(''.join(_y.split())) + ['END'])
        x.append(start + ''.join(jieba.cut(''.join(_x.split()))) + end)
        y.append(start + ''.join(jieba.cut(''.join(_y.split()))) + end)
        c += 1
        if c == maxlen:
            break
    return x, y


class Word2Word:
    def __init__(self):
        self.encoder_input_data = []
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

        x, y = load_cha_cha(samples)

        for _x, _y in zip(x, y):
            for cx in _x:
                input_characters.add(cx)
            for cy in _y:
                target_characters.add(cy)

        self.max_input_seq_length = input_max_len
        self.max_output_seq_length = output_max_len

        self.len_input_characters = len(input_characters) + 2
        self.len_output_characters = len(target_characters) + 1

        print('输入字符集长度', self.len_input_characters)
        print('输出字符集长度', self.len_output_characters)
        print('输入字符串最大长度', self.max_input_seq_length)
        print('输出字符串最大长度', self.max_output_seq_length)

        self.input_char_index = {char: i + 2 for i, char in enumerate(sorted(list(input_characters)))}
        self.output_char_index = {char: i + 1 for i, char in enumerate(sorted(list(target_characters)))}

        self.input_char_index['PAD'] = 0
        self.input_char_index['UNK'] = 1
        self.output_char_index['UNK'] = 0

        for input_text, target_text in zip(x, y):
            encoder_input_wids = []
            for w in input_text:
                w2id = 1
                if w in self.input_char_index:
                    w2id = self.input_char_index[w]
                encoder_input_wids.append(w2id)
            self.encoder_input_data.append(encoder_input_wids)

        self.encoder_input_data = pad_sequences(self.encoder_input_data, self.max_input_seq_length)
        self.decoder_input_data = np.zeros(shape=(len(x), self.max_output_seq_length, self.len_output_characters))
        self.decoder_output_data = np.zeros(shape=(len(x), self.max_output_seq_length, self.len_output_characters))

        for index, output_text in enumerate(y):
            for idx, w in enumerate(output_text):
                if idx >= self.max_output_seq_length:
                    break
                w2id = 0
                if w in self.output_char_index:
                    w2id = self.output_char_index[w]
                self.decoder_input_data[index, idx, w2id] = 1.0
                if idx > 0:
                    self.decoder_output_data[index, idx - 1, w2id] = 1.0

    def build_model(self):
        # 编码器 = 输入层 + LSTM层
        encoder_inputs = layers.Input(shape=(None,), name='encoder_inputs')
        encoder_embedding = layers.Embedding(input_dim=self.len_input_characters, output_dim=embedding_dim,
                                             input_length=self.max_input_seq_length, name='encoder_embedding')
        encoder_lstm = layers.LSTM(latent_dim, return_state=True, kernel_initializer=he_normal(1), name='encoder_lstm')
        encoder_outputs, state_h, state_c = encoder_lstm(encoder_embedding(encoder_inputs))
        encoder_state = [state_h, state_c]

        # 解码器 = 输出层 + LSTM层 + Dense层
        decoder_inputs = layers.Input(shape=(None, self.len_output_characters), name='decoder_inputs')
        decoder_lstm = layers.LSTM(latent_dim, return_state=True, return_sequences=True,
                                   kernel_initializer=he_normal(2), name='decoder_lstm')
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
        self.save(model, seq2seq2_model_path, word2word_obj_path)

    def load_model(self, seq2seq_model_path):
        model = models.load_model(seq2seq_model_path)
        encoder_inputs = model.get_layer('encoder_inputs').input
        encoder_embedding = model.get_layer('encoder_embedding')
        encoder_lstm = model.get_layer('encoder_lstm')
        encoder_outputs, i_state_h, i_state_c = encoder_lstm(encoder_embedding(encoder_inputs))
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
            input_w2id = []
            # for index, word in enumerate(jieba.cut(sentence)):
            for index, word in enumerate(sentence):
                idx = 1
                if word in self.input_char_index:
                    idx = self.input_char_index[word]
                input_w2id.append(idx)
            input_vector = pad_sequences([input_w2id], maxlen=self.max_input_seq_length)

            decoded_sentence = self.decode_sequence(input_vector, reverse_target_char_index)
            return decoded_sentence

        self.load(seq2seq2_model_path, word2word_obj_path)
        reverse_target_char_index = {i: char for char, i in self.output_char_index.items()}
        while True:
            sentence = input('question:')
            retv = input_sentence_vector(self, sentence)
            print('answer:', retv)

    def decode_sequence(self, input_seq, reverse_target_char_index):
        states_value = self.encoder_model.predict(input_seq)
        target_seq = np.zeros(shape=(1, 1, self.len_output_characters))
        target_seq[0, 0, self.output_char_index[start]] = 1.

        stop = False
        decoded_sentence = ''
        decoded_sentence_len = 0
        while not stop:
            output_tokens, h, c = self.decoder_model.predict([target_seq] + states_value)
            sampled_token_index = np.argmax(output_tokens[0, -1, :])

            sampled_char = reverse_target_char_index[sampled_token_index]

            if sampled_char != start and sampled_char != end:
                decoded_sentence += ' ' + sampled_char

            decoded_sentence_len += 1
            if sampled_char == end or decoded_sentence_len > self.max_output_seq_length:
                stop = True

            target_seq = np.zeros(shape=(1, 1, self.len_output_characters))
            target_seq[0, 0, sampled_token_index] = 1.

            states_value = [h, c]
        return decoded_sentence.strip()


if __name__ == '__main__':
    # load_cha_cha(1)
    eng = Word2Word()
    eng.train()
    # eng.predict()
