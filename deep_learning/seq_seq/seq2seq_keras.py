import seq2seq
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from seq2seq.models import SimpleSeq2Seq, Seq2Seq

from corpus.load_corpus import LoadCorpus

max_words = 15
hidden_dim = 256
word_dim = 60
ty = 10

batch_size = 32
epochs = 100


def test():
    x, y = LoadCorpus.load_chatbot100_train()
    # x, y = LoadCorpus.load_xiaohuangji_train()
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
    # model = SimpleSeq2Seq(input_dim=5, hidden_dim=10, output_length=8, output_dim=8, depth=3)
    model = Seq2Seq(batch_input_shape=(None, 15, 60), hidden_dim=256, output_length=15, output_dim=60, depth=3)
    model.compile(loss='mse', optimizer='rmsprop', metrics=['accuracy'])
    print(model.summary())

    model.fit(train_x, y_input, epochs=epochs, batch_size=8)


if __name__ == '__main__':
    test()
