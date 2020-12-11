import os

import numpy as np
from keras import models, layers
from keras.initializers import he_normal
from keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.metrics import f1_score
from gensim.models import KeyedVectors

from corpus import tianchi_news_class_path
from tools import running_of_time
from deep_learning.simple_attention.simple_attention2_keras import Attention
from tianchi.news_classifier import build_word2vec


class BaseNN:
    def __init__(self, *args, **kwargs):
        self.vocab_size = kwargs.get('vocab_size')
        self.word_vocab = kwargs.get('word_vocab')
        self.embedding_dim = kwargs.get('embedding_dim')
        self.max_words = kwargs.get('max_words')
        self.class_num = kwargs.get('class_num')
        self.epochs = kwargs.get('epochs')
        self.batch_size = kwargs.get('batch_size')
        self.m = None
        self.model_name = None

    @running_of_time
    def build_myself_embedding_matrix(self):
        embedding_dim = 400
        word2vec_model = build_word2vec()
        glove_model = KeyedVectors.load_word2vec_format(os.path.join(tianchi_news_class_path, 'glove_200.txt'),
                                                        binary=False)
        embedding_matrix = np.zeros(shape=(self.vocab_size, embedding_dim))

        unknow_vec = np.random.random(embedding_dim) * 0.5
        unknow_vec -= unknow_vec.mean()
        for word, idx in self.word_vocab.items():
            if word in word2vec_model:
                embedding_matrix[idx] = np.concatenate((word2vec_model.wv[word], glove_model.wv[word]))
            else:
                embedding_matrix[idx] = unknow_vec
        print('matrix shape', embedding_matrix.shape)
        return embedding_matrix

    def evaluate(self, test_x, test_y):
        loss, acc = self.m.evaluate(x=test_x, y=test_y, batch_size=1024)
        print('acc', acc, 'loss', loss)

    def f1_score(self, y_true, y_pred):
        y_true = y_true.reshape(-1, )
        y_pred = y_pred.reshape(-1, )
        if len(set(y_true)) > 2:
            score_f1 = f1_score(y_true, y_pred, average='macro')
        else:
            score_f1 = f1_score(y_true, y_pred)
        print('f1 score', score_f1)

    def save_model(self):
        self.m.save(os.path.join(tianchi_news_class_path, self.model_name))

    def load_model(self):
        self.m = models.load_model(os.path.join(tianchi_news_class_path, 'textcnn_0.9449.model'))
        # self.m = models.load_model(os.path.join(tianchi_news_class_path, self.model_name))

    def callbacks(self):
        callback = ModelCheckpoint(filepath=os.path.join(tianchi_news_class_path, self.model_name), monitor='roc_auc',
                                   verbose=1, save_best_only=True, mode='max', period=2)
        earlystop = EarlyStopping(monitor='val_acc', patience=2, verbose=1, mode='max')
        return [callback, earlystop]


class FastText(BaseNN):
    def __init__(self, *args, **kwargs):
        super(FastText, self).__init__(*args, **kwargs)
        self.model_name = 'fasttext_{val_acc:.4f}.model'

    def model(self):
        model = models.Sequential()
        model.add(layers.Embedding(self.vocab_size + 1, self.embedding_dim, input_length=self.max_words))
        # model.add(layers.GlobalAveragePooling1D())
        model.add(layers.GlobalMaxPooling1D())
        model.add(layers.Dense(self.class_num, activation='softmax', kernel_initializer=he_normal()))
        print(model.summary())
        return model

    @running_of_time
    def train(self, train_x, train_y):
        self.m = self.model()
        self.m.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        self.m.fit(train_x, train_y, epochs=self.epochs, batch_size=self.batch_size, validation_split=0.2,
                   callbacks=self.callbacks(), shuffle=True)
        # self.save_model()


class TextCNN(BaseNN):
    def __init__(self, *args, **kwargs):
        super(TextCNN, self).__init__(*args, **kwargs)
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

        conv_3 = layers.Conv2D(256, (2, self.embedding_dim), padding='valid', activation='relu')(reshape)
        maxpool_3 = layers.MaxPool2D((self.max_words - 2 + 1, 1), strides=(1, 1), padding='valid')(conv_3)

        con_cat = layers.concatenate(inputs=[maxpool_3, maxpool_0, maxpool_1, maxpool_2], axis=1)
        print('concat shape', con_cat.shape)
        flatten = layers.Flatten()(con_cat)
        print('flatten shape', flatten.shape)
        # atten = SimpleAttention()(con_cat)
        # flat = layers.Flatten()(atten)

        # attention = Attention(step_dim=4)(con_cat)
        # flatten = layers.Flatten()(attention)

        drop = layers.Dropout(0.4)(flatten)
        output = layers.Dense(self.class_num, activation='softmax')(drop)
        model = models.Model(inputs=inputs, output=output)
        print(model.summary())
        return model

    def model_3(self):
        embedding_matrix = self.build_myself_embedding_matrix()
        inputs = layers.Input(shape=(self.max_words,), dtype='int32')
        print(self.embedding_dim)
        embedding = layers.Embedding(input_dim=self.vocab_size, output_dim=self.embedding_dim,
                                     input_length=self.max_words, weights=[embedding_matrix])
        print('embedding', embedding)
        embed = embedding(inputs)
        embed = layers.SpatialDropout1D(0.3)(embed)
        convs = []
        for kernel_size in [2, 3, 4, 5]:
            c = layers.Conv1D(1024, kernel_size, activation='relu')(embed)
            c = layers.GlobalMaxPool1D()(c)
            convs.append(c)
        x = layers.Concatenate()(convs)
        x = layers.Dense(512, activation='relu')(x)
        x = layers.Dropout(0.4)(x)
        output = layers.Dense(self.class_num, activation='softmax')(x)
        model = models.Model(input=inputs, output=output)
        print(model.summary())
        return model

    @running_of_time
    def train(self, train_x, train_y, test_x=None, test_y=None):
        self.m = self.model_3()
        self.m.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        self.m.fit(train_x, train_y, epochs=self.epochs, batch_size=self.batch_size, validation_data=(test_x, test_y),
                   verbose=2, callbacks=self.callbacks(), shuffle=True)

    def predict(self, test_x):
        r = self.m.predict(test_x, batch_size=1024, verbose=1)
        print(r.shape)
        print(r[0])


class TextRNN(BaseNN):
    def __init__(self, *args, **kwargs):
        super(TextRNN, self).__init__(*args, **kwargs)
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

        atten = Attention(self.max_words)(lstm)
        print('atten', atten.shape)

        output = layers.Dense(self.class_num, activation='softmax')(atten)
        model = models.Model(input=word_input, output=output)
        print(model.summary())
        return model

    def model_3(self):
        embedding_matrix = self.build_myself_embedding_matrix()
        input = layers.Input(shape=(self.max_words,))
        embedding = layers.Embedding(input_dim=embedding_matrix.shape[0], output_dim=embedding_matrix.shape[1],
                                     input_length=self.max_words, weights=[embedding_matrix])
        x = layers.SpatialDropout1D(0.2)(embedding(input))
        x = layers.Bidirectional(layers.GRU(400, return_sequences=True))(x)
        x = layers.Bidirectional(layers.GRU(400, return_sequences=True))(x)
        avg_pool = layers.GlobalAveragePooling1D()(x)
        max_pool = layers.GlobalMaxPool1D()(x)
        concat = layers.concatenate([avg_pool, max_pool])

        x = layers.Dense(1024)(concat)
        x = layers.BatchNormalization()(x)
        x = layers.Activation(activation='relu')(x)
        x = layers.Dropout(0.2)(x)

        x = layers.Dense(512)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation(activation='relu')(x)
        x = layers.Dropout(0.2)(x)
        output = layers.Dense(self.class_num, activation='softmax')(x)

        model = models.Model(input=input, output=output)
        print(model.summary())
        return model

    @running_of_time
    def train(self, train_x, train_y):
        self.m = self.model_3()
        self.m.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        self.m.fit(train_x, train_y, epochs=self.epochs, batch_size=self.batch_size, validation_split=0.2, verbose=1,
                   callbacks=self.callbacks(), shuffle=True)
