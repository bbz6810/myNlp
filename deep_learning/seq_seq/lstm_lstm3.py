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

import numpy as np
from sklearn.model_selection import train_test_split
from keras import layers, models

from corpus import chinese_to_english_path, french_to_english_path, seq2seq2_model_path
from tools.chinese_trans.langconv import cht_to_chs
from corpus.load_corpus import LoadCorpus

batch_size = 64
epochs = 10
latent_dim = 256
samples = 10000


def load_seq(maxlen):
    x, y = [], []
    with open(chinese_to_english_path, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            line = line.strip().split('\t')
            x.append(line[0])
            y.append('\t' + cht_to_chs(line[1]) + '\n')
    print('对话语料大小', len(x))
    return x[:maxlen], y[:maxlen]


def load_seq2(maxlen):
    x, y = [], []
    with open(french_to_english_path, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            line = line.strip().split('\t')
            x.append(line[0])
            y.append('\t' + line[1] + '\n')
    print('对话语料大小', len(x))
    return x[:maxlen], y[:maxlen]


def load_seq3(maxlen):
    x, y = LoadCorpus.load_chatbot100_train()
    # x, y = LoadCorpus.load_xiaohuangji_train()
    x = [''.join(s.split()) for s in x]
    y = [''.join(s.split()) for s in y]
    y = ['\t' + s + '\n' for s in y]
    return x[:maxlen], y[:maxlen]


input_characters = set()
target_characters = set()


def run():
    x, y = load_seq3(samples)
    for _x, _y in zip(x, y):
        for cx in _x:
            input_characters.add(cx)
        for cy in _y:
            target_characters.add(cy)

    max_input_seq_length = max(len(t) for t in x)
    max_output_seq_length = max(len(t) for t in y)

    print('输入字符集', len(input_characters))
    print('输出字符集', len(target_characters))
    print('输入字符串最大长度', max_input_seq_length)
    print('输出字符串最大长度', max_output_seq_length)

    input_token_index = {char: i for i, char in enumerate(sorted(list(input_characters)))}
    target_token_index = {char: i for i, char in enumerate(sorted(list(target_characters)))}

    encoder_input_data = np.zeros(shape=(len(x), max_input_seq_length, len(input_characters)), dtype='float32')
    decoder_input_data = np.zeros(shape=(len(x), max_output_seq_length, len(target_characters)), dtype='float32')
    decoder_target_data = np.zeros(shape=(len(x), max_output_seq_length, len(target_characters)), dtype='float32')

    print('encoder 输入.shape', encoder_input_data.shape)
    print('decoder 输入.shape', decoder_input_data.shape)
    print('decoder 输出.shape', decoder_target_data.shape)

    # for i, j in zip(x, y):
    #     print(i, j)
    # return

    for i, (input_text, target_text) in enumerate(zip(x, y)):
        for t, char in enumerate(input_text):
            encoder_input_data[i, t, input_token_index[char]] = 1.0
        for t, char in enumerate(target_text):
            decoder_input_data[i, t, target_token_index[char]] = 1.0
            if t > 0:
                decoder_target_data[i, t - 1, target_token_index[char]] = 1.0

    # 生成嵌入式向量
    wv_model = LoadCorpus.load_wv_model()
    embedding_matrix = np.zeros(shape=(len(input_token_index) + 1, 60))
    for word, index in input_token_index.items():
        try:
            embedding_matrix[index] = wv_model[word]
        except Exception as e:
            pass

    # 编码器
    encoder_inputs = layers.Input(shape=(None, len(input_characters)), name='encoder_inputs')
    encoder_lstm = layers.LSTM(latent_dim, return_state=True, name='encoder_lstm')
    encoder_outputs, state_h, state_c = encoder_lstm(encoder_inputs)
    encoder_state = [state_h, state_c]

    # 解码器
    decoder_inputs = layers.Input(shape=(None, len(target_characters)), name='decoder_inputs')
    decoder_lstm = layers.LSTM(latent_dim, return_state=True, return_sequences=True, name='decoder_lstm')
    decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_state)
    decoder_dense = layers.Dense(len(target_characters), activation='softmax', name='decoder_dense')
    decoder_outputs = decoder_dense(decoder_outputs)

    model = models.Model(inputs=[encoder_inputs, decoder_inputs], outputs=decoder_outputs)
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(x=[encoder_input_data, decoder_input_data], y=decoder_target_data, batch_size=batch_size, epochs=epochs,
              validation_split=0.2)
    model.save(seq2seq2_model_path)

    model = models.load_model(seq2seq2_model_path)
    # encoder_inputs = model.get_layer('encoder_inputs').input
    encoder_inputs = layers.Input(shape=(None, len(input_characters)), name='encoder_inputs')
    encoder_lstm = model.get_layer('encoder_lstm')
    encoder_outputs, i_state_h, i_state_c = encoder_lstm(encoder_inputs)
    encoder_state = [i_state_h, i_state_c]

    encoder_model = models.Model(encoder_inputs, encoder_state)

    # decoder_inputs = model.get_layer('decoder_inputs').input
    decoder_inputs = layers.Input(shape=(None, len(target_characters)), name='decoder_inputs')
    decoder_lstm = model.get_layer('decoder_lstm')
    decoder_dense = model.get_layer('decoder_dense')

    decoder_state_input_h = layers.Input(shape=(latent_dim,))
    decoder_state_input_c = layers.Input(shape=(latent_dim,))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

    decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)
    decoder_states = [state_h, state_c]
    decoder_outputs = decoder_dense(decoder_outputs)
    decoder_model = models.Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)

    reverse_input_char_index = {i: char for char, i in input_token_index.items()}
    reverse_target_char_index = {i: char for char, i in target_token_index.items()}

    def decode_sequence(input_seq):
        states_value = encoder_model.predict([input_seq])

        target_seq = np.zeros(shape=(1, 1, len(target_characters)))
        target_seq[0, 0, target_token_index['\t']] = 1.

        stop = False
        decoded_sentence = ''

        while not stop:
            output_tokens, h, c = decoder_model.predict([target_seq] + states_value)

            sampled_token_index = np.argmax(output_tokens[0, -1, :])
            sampled_char = reverse_target_char_index[sampled_token_index]
            decoded_sentence += sampled_char

            if sampled_char == '\n' or len(decoded_sentence) > max_output_seq_length:
                stop = True

            target_seq = np.zeros(shape=(1, 1, len(target_characters)))
            target_seq[0, 0, sampled_token_index] = 1.

            states_value = [h, c]
        return decoded_sentence

    c = 0

    for seq_index in range(samples):
        input_seq = encoder_input_data[seq_index:seq_index + 1]
        decoded_sentence = decode_sequence(input_seq)
        print('输入序列', x[seq_index].strip(), '输出序列', y[seq_index].strip(), '预测序列', decoded_sentence)
        if y[seq_index].strip() == decoded_sentence.strip():
            c += 1
    print('正确个数', c, '正确率', c / samples)


if __name__ == '__main__':
    run()
    # load_seq3(1)
