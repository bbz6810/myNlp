import numpy as np
from keras import models, layers
from deep_learning.data_pretreatment import Pretreatment
from deep_learning.nn import NN

batch_size = 1024
epochs = 3
lstm_dim = 128


class TextRNN(NN):
    def __init__(self, nn_param):
        super(TextRNN, self).__init__(nn_param=nn_param)

    def model(self, embedding_matrix):
        inputs = layers.Input(shape=(self.nn_param.max_words,))
        wv = layers.Embedding(self.nn_param.vocab_size + 1, self.nn_param.embedding_dim, weights=[embedding_matrix])(
            inputs)
        lstm = layers.LSTM(lstm_dim)(wv)
        if self.nn_param.class_num == 2:
            y = layers.Dense(1, activation='sigmoid')(lstm)
        else:
            y = layers.Dense(self.nn_param.class_num, activation='softmax')(lstm)
        model = models.Model(input=inputs, output=y)
        print(model.summary())
        return model

    def train(self, train_x, train_y, embedding_matrix):
        self.nn = self.model(embedding_matrix)
        if self.nn_param.class_num == 2:
            self.nn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        else:
            self.nn.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        self.nn.fit(train_x, train_y, epochs=epochs, batch_size=batch_size, validation_split=0.15)

    def predict(self, test_x, test_y):
        _y = self.nn.predict(test_x)
        self.score(_y, test_y)


def run():
    pretreatment = Pretreatment()
    train_x, test_x, train_y, test_y = pretreatment.train_test_split(c=3, test_size=0.6)
    embedding_matrix = pretreatment.create_embedding_matrix(15000)
    textrnn = TextRNN(pretreatment.nnparam)
    textrnn.train(train_x, train_y, embedding_matrix)
    textrnn.predict(test_x, test_y)


if __name__ == '__main__':
    run()
