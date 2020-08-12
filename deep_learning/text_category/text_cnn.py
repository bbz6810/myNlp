from keras import models, layers

from deep_learning.nn import NN
from deep_learning.data_pretreatment import Pretreatment

batch_size = 64
epochs = 3


class TextCNN(NN):
    def __init__(self, nn_param):
        super(TextCNN, self).__init__(nn_param=nn_param)

    def model(self, embedding_matrix):
        inputs = layers.Input(shape=(self.nn_param.max_words,), dtype='float32')
        embedding = layers.Embedding(input_dim=self.nn_param.vocab_size + 1, output_dim=self.nn_param.embedding_dim,
                                     input_length=self.nn_param.max_words, trainable=False, weights=[embedding_matrix])
        embed = embedding(inputs)

        cnn1 = layers.Conv1D(256, 3, padding='same', strides=1, activation='relu')(embed)
        cnn1 = layers.MaxPool1D(pool_size=48)(cnn1)

        cnn2 = layers.Conv1D(256, 4, padding='same', strides=1, activation='relu')(embed)
        cnn2 = layers.MaxPool1D(pool_size=47)(cnn2)

        cnn3 = layers.Conv1D(256, 5, padding='same', strides=1, activation='relu')(embed)
        cnn3 = layers.MaxPool1D(pool_size=46)(cnn3)

        cnn = layers.concatenate(inputs=[cnn1, cnn2, cnn3], axis=1)
        flat = layers.Flatten()(cnn)
        drop = layers.Dropout(0.2)(flat)
        if self.nn_param.class_num == 2:
            output = layers.Dense(1, activation='sigmoid')(drop)
        else:
            output = layers.Dense(self.nn_param.class_num, activation='sigmoid')(drop)
        model = models.Model(inputs=inputs, outputs=output)
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
    train_x, test_x, train_y, test_y = pretreatment.train_test_split(c=5, test_size=0.6)
    embedding_matrix = pretreatment.create_embedding_matrix(20000)
    textrnn = TextCNN(pretreatment.nnparam)
    textrnn.train(train_x, train_y, embedding_matrix)
    textrnn.predict(test_x, test_y)
    """
    16178/16178 [==============================] - 100s 6ms/step - loss: 0.0672 - acc: 0.9794 - val_loss: 0.2334 - val_acc: 0.9370
    正确个数 26730 总数 28551 正确率 0.9362193968687612
    """


if __name__ == '__main__':
    run()
