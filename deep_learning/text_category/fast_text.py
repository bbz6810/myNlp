from keras import models, layers

from deep_learning.nn import NN
from deep_learning.data_pretreatment import Pretreatment

batch_size = 64
epochs = 5


class FastText(NN):
    def __init__(self, nn_param):
        super(FastText, self).__init__(nn_param=nn_param)

    def model(self, embedding_matrix):
        model = models.Sequential()
        model.add(layers.Embedding(self.nn_param.vocab_size + 1, self.nn_param.embedding_dim,
                                   input_length=self.nn_param.max_words, weights=[embedding_matrix]))
        model.add(layers.GlobalAveragePooling1D())
        if self.nn_param.class_num == 2:
            model.add(layers.Dense(1, activation='sigmoid'))
            model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        else:
            model.add(layers.Dense(self.nn_param.class_num, activation='softmax'))
            model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        print(model.summary())
        return model

    def train(self, train_x, train_y, embedding_matrix):
        self.nn = self.model(embedding_matrix)
        self.nn.fit(train_x, train_y, epochs=epochs, batch_size=batch_size, validation_split=0.2)

    def predict(self, test_x, test_y):
        _y = self.nn.predict(test_x)
        self.score(_y, test_y)


def run():
    pretreatment = Pretreatment()
    train_x, test_x, train_y, test_y = pretreatment.train_test_split(c=5, test_size=0.6)
    embedding_matrix = pretreatment.create_embedding_matrix(20000)
    textrnn = FastText(pretreatment.nnparam)
    textrnn.train(train_x, train_y, embedding_matrix)
    textrnn.predict(test_x, test_y)


if __name__ == '__main__':
    run()
