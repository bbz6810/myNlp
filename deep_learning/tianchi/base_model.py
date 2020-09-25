import os

from keras import models, layers
from keras.initializers import he_normal
from keras.callbacks import ModelCheckpoint

from corpus import tianchi_news_class_path
from tools import running_of_time
from deep_learning.simple_attention.simple_attention_keras import SimpleAttention


class BaseNN:
    def __init__(self, vocab_size, embedding_dim, max_words, class_num, epochs, batch_size):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.max_words = max_words
        self.class_num = class_num
        self.epochs = epochs
        self.batch_size = batch_size
        self.m = None
        self.model_name = None

    def predict(self, test_x, test_y):
        loss, acc = self.m.evaluate(x=test_x, y=test_y, batch_size=1024)
        print('acc', acc, 'loss', loss)

    def save_model(self):
        self.m.save(os.path.join(tianchi_news_class_path, self.model_name))

    def load_model(self):
        self.m = models.load_model(os.path.join(tianchi_news_class_path, self.model_name))

    def callbacks(self):
        callback = ModelCheckpoint(filepath=os.path.join(tianchi_news_class_path, self.model_name), monitor='val_acc',
                                   verbose=1, save_best_only=True, mode='max', period=2)
        return [callback]


class FastText(BaseNN):
    def __init__(self, vocab_size, embedding_dim, max_words, class_num, epochs, batch_size):
        super(FastText, self).__init__(vocab_size, embedding_dim, max_words, class_num, epochs, batch_size)
        self.model_name = 'fasttext_{val_acc:.4f}.model'

    def model(self):
        model = models.Sequential()
        model.add(layers.Embedding(self.vocab_size + 1, self.embedding_dim, input_length=self.max_words))
        # model.add(layers.GlobalAveragePooling1D())
        model.add(layers.GlobalMaxPooling1D())
        model.add(layers.Dense(self.class_num, activation='softmax', kernel_initializer=he_normal()))
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        print(model.summary())
        return model

    @running_of_time
    def train(self, train_x, train_y):
        self.m = self.model()
        self.m.fit(train_x, train_y, epochs=self.epochs, batch_size=self.batch_size, validation_split=0.2,
                   callbacks=self.callbacks(), shuffle=True)
        # self.save_model()


class TextCNN(BaseNN):
    def __init__(self, vocab_size, embedding_dim, max_words, class_num, epochs, batch_size):
        super(TextCNN, self).__init__(vocab_size, embedding_dim, max_words, class_num, epochs, batch_size)
        self.model_name = 'textcnn_{val_acc:.4f}.model'

    def model_1(self):
        inputs = layers.Input(shape=(self.max_words,), dtype='float32')
        embedding = layers.Embedding(input_dim=self.vocab_size + 1, output_dim=self.embedding_dim,
                                     input_length=self.max_words)
        embed = embedding(inputs)
        # 907*100 - 907*256 -
        cnn1 = layers.Conv1D(256, 3, padding='same', strides=1, activation='relu')(embed)
        cnn1 = layers.MaxPool1D(pool_size=48)(cnn1)

        cnn2 = layers.Conv1D(256, 4, padding='same', strides=1, activation='relu')(embed)
        cnn2 = layers.MaxPool1D(pool_size=47)(cnn2)

        cnn3 = layers.Conv1D(256, 5, padding='same', strides=1, activation='relu')(embed)
        cnn3 = layers.MaxPool1D(pool_size=46)(cnn3)

        cnn_con = layers.concatenate(inputs=[cnn1, cnn2, cnn3], axis=1)

        flat = layers.Flatten()(cnn_con)
        drop = layers.Dropout(0.5)(flat)
        output = layers.Dense(self.class_num, activation='softmax')(drop)
        model = models.Model(inputs=inputs, outputs=output)
        print(model.summary())
        return model

    def model_2(self):
        inputs = layers.Input(shape=(self.max_words,))
        embedding = layers.Embedding(input_dim=self.vocab_size + 1, output_dim=self.embedding_dim,
                                     input_length=self.max_words)(inputs)
        reshape = layers.Reshape((self.max_words, self.embedding_dim, 1))(embedding)

        conv_0 = layers.Conv2D(256, (3, self.embedding_dim), padding='valid', activation='relu')(reshape)
        maxpool_0 = layers.MaxPool2D((self.max_words - 3 + 1, 1), strides=(1, 1), padding='valid')(conv_0)

        conv_1 = layers.Conv2D(256, (4, self.embedding_dim), padding='valid', activation='relu')(reshape)
        maxpool_1 = layers.MaxPool2D((self.max_words - 4 + 1, 1), strides=(1, 1), padding='valid')(conv_1)

        conv_2 = layers.Conv2D(256, (5, self.embedding_dim), padding='valid', activation='relu')(reshape)
        maxpool_2 = layers.MaxPool2D((self.max_words - 5 + 1, 1), strides=(1, 1), padding='valid')(conv_2)

        con_cat = layers.concatenate(inputs=[maxpool_0, maxpool_1, maxpool_2], axis=1)
        flatten = layers.Flatten()(con_cat)
        # print('flatten shape', flatten.shape)
        # atten = SimpleAttention()(con_cat)
        # flat = layers.Flatten()(atten)

        drop = layers.Dropout(0.4)(flatten)

        output = layers.Dense(self.class_num, activation='softmax')(drop)

        model = models.Model(inputs=inputs, output=output)
        print(model.summary())
        return model

    @running_of_time
    def train(self, train_x, train_y):
        self.m = self.model_2()
        self.m.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        self.m.fit(train_x, train_y, epochs=self.epochs, batch_size=self.batch_size, validation_split=0.2, verbose=2,
                   callbacks=self.callbacks(), shuffle=True)


class TextRNN(BaseNN):
    def __init__(self, vocab_size, embedding_dim, max_words, class_num, epochs, batch_size):
        super(TextRNN, self).__init__(vocab_size, embedding_dim, max_words, class_num, epochs, batch_size)
        self.model_name = 'textrnn_{val_acc:.4f}.model'

    def model_1(self):
        inputs = layers.Input(shape=(self.max_words,))
        embedding = layers.Embedding(input_dim=self.vocab_size + 1, output_dim=self.embedding_dim)(inputs)
        lstm = layers.LSTM(128)(embedding)
        y = layers.Dense(self.class_num, activation='softmax')(lstm)
        model = models.Model(input=inputs, output=y)
        print(model.summary())
        return model

    def model_2(self):
        word_input = layers.Input(shape=(self.max_words,))
        word_embeds = layers.Embedding(input_dim=self.vocab_size + 1, output_dim=self.embedding_dim)(word_input)

        lstm = layers.Bidirectional(layers.LSTM(units=128, return_sequences=True))(word_embeds)
        print('lstm shape', lstm)

        atten = SimpleAttention()(lstm)
        print('atten', atten.shape)

        output = layers.Dense(self.class_num, activation='softmax')(atten)
        model = models.Model(input=word_input, output=output)
        print(model.summary())
        return model

    @running_of_time
    def train(self, train_x, train_y):
        self.m = self.model_2()
        self.m.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        self.m.fit(train_x, train_y, epochs=self.epochs, batch_size=self.batch_size, validation_split=0.2, verbose=1,
                   callbacks=self.callbacks(), shuffle=True)
        self.save_model()
