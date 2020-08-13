import numpy as np
import jieba
from keras import layers, models
from keras import optimizers
from keras.preprocessing.sequence import pad_sequences
from corpus import xiaohuangji_model_path
from corpus.load_corpus import LoadCorpus

max_words = 15
hidden_dim = 256
word_dim = 60
ty = 10

batch_size = 32
epochs = 100


def train():
    # x, y = LoadCorpus.load_chatbot100_train()
    x, y = LoadCorpus.load_xiaohuangji_train()
    wv = LoadCorpus.load_wv60_model()

    def trans_seq(d):
        vector = [[wv[c] for c in s.split(' ') if c in wv.wv.vocab] for s in d]
        t = pad_sequences(vector, maxlen=max_words, padding='post', value=1., dtype='float32')
        return t

    def generate_decoder_input(decoder_output):
        word_dim = len(decoder_output[0][0])
        word_start = np.zeros(shape=(word_dim,))
        decoder_input = []
        if not (decoder_input is decoder_output):
            for example in decoder_output:
                t = list(example[:14])
                t.insert(0, word_start)
                decoder_input.append(np.array(t))
        decoder_input = pad_sequences(decoder_input, maxlen=15, dtype='float32')
        return decoder_input

    train_x = trans_seq(x)
    train_y = trans_seq(y)
    y_input = generate_decoder_input(train_y)

    model = build_nn()
    model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
    model.fit([train_x, y_input], train_y, epochs=epochs, batch_size=batch_size)
    model.save(xiaohuangji_model_path)


def build_nn():
    encoder_input = layers.Input(shape=(None, word_dim), name='encoder_input')
    encoder_lstm = layers.LSTM(hidden_dim, return_state=True, name='encoder_lstm')
    encoder_output, state_h, state_c = encoder_lstm(encoder_input)
    encoder_state = [state_h, state_c]

    decoder_input = layers.Input(shape=(None, word_dim), name='decoder_input')
    decoder_lstm = layers.LSTM(hidden_dim, return_state=True, return_sequences=True, name='decoder_lstm')
    decoder_output, _, _ = decoder_lstm(decoder_input, initial_state=encoder_state)
    decoder_dense = layers.TimeDistributed(layers.Dense(output_dim=word_dim, activation='linear'), name='densor')
    outputs = decoder_dense(decoder_output)

    train_model = models.Model(inputs=[encoder_input, decoder_input], outputs=outputs)
    print(train_model.summary())
    return train_model


def predict_model(encoder_layer, decoder_layer, time_densor, word_dim, ty):
    x0 = layers.Input(shape=(None, word_dim), name='sentence_input')
    _, state_h, state_c = encoder_layer(x0)
    decoder_state_inputs = [state_h, state_c]

    decoder_input = layers.Input(shape=(1, word_dim), name='decoder_initial_input')
    x = decoder_input
    outputs = []
    for i in range(ty):
        decoder_output, h, c = decoder_layer(x, initial_state=decoder_state_inputs)
        output = time_densor(decoder_output)
        decoder_state_inputs = [h, c]
        x = output
        outputs.append(output)

    model = models.Model(input=[x0, decoder_input], outputs=outputs)
    return model


def input_sentence_vector(sentence, word2vec_model):
    sentence = sentence.strip()
    word_vector = [word2vec_model[w] for w in jieba.cut(sentence) if w in word2vec_model.wv.vocab]

    word_end = np.ones(shape=(word_dim,), dtype='float32')
    if len(word_vector) > max_words - 1:
        word_vector[max_words:] = []
        word_vector.append(word_end)
    else:
        for i in range(max_words - len(word_vector)):
            word_vector.append(word_end)
    return np.array([word_vector], dtype='float32')


def vector_sentence(answer_sequence, word2vec_model):
    answer_list = [word2vec_model.most_similar([answer_sequence[i][0][0]])[0] for i in range(ty)]
    answer = ''
    for index, word_tuple in enumerate(answer_list):
        if word_tuple[1] > 0.75:
            answer += str(word_tuple[0])
    return answer


def predict():
    model = models.load_model(xiaohuangji_model_path)
    encoder_lstm = model.get_layer('encoder_lstm')
    decoder_lstm = model.get_layer('decoder_lstm')
    densor = model.get_layer('densor')
    decoder_model = predict_model(encoder_lstm, decoder_lstm, densor, word_dim, ty)
    wv = LoadCorpus.load_wv60_model()
    while True:
        sentence = input('问:')
        sentence_vec = input_sentence_vector(sentence, wv)
        print(sentence_vec.shape)
        x = np.zeros(shape=(1, 1, word_dim))
        answer_sequence = decoder_model.predict([sentence_vec, x])
        print('答:', vector_sentence(answer_sequence, word2vec_model=wv))


if __name__ == '__main__':
    # predict()
    # train()
    build_nn()