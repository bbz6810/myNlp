# https://github.com/ShawnyXiao/TextClassification-Keras/blob/master/model/HAN/attention.py

import keras.backend as K
from keras import layers, initializers, regularizers, constraints


class Attention(layers.Layer):
    def __init__(self, step_dim, w_regularizer=None, b_regularizer=None, w_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
        """
        Keras Layer that implements an Attention mechanism for temporal data.
        Supports Masking.
        Follows the work of Raffel et al. [https://arxiv.org/abs/1512.08756]
        # Input shape
            3D tensor with shape: `(samples, steps, features)`.
        # Output shape
            2D tensor with shape: `(samples, features)`.
        :param kwargs:
        Just put it on top of an RNN Layer (GRU/LSTM/SimpleRNN) with return_sequences=True.
        The dimensions are inferred based on the output shape of the RNN.
        Example:
            # 1
            model.add(LSTM(64, return_sequences=True))
            model.add(Attention())
            # next add a Dense layer (for classification/regression) or whatever...
            # 2
            hidden = LSTM(64, return_sequences=True)(words)
            sentence = Attention()(hidden)
            # next add a Dense layer (for classification/regression) or whatever...
        """

        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')

        self.w_regularizer = regularizers.get(w_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.w_constraint = constraints.get(w_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        self.step_dim = step_dim
        self.features_dim = 0

        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.w = self.add_weight(name='{}_w'.format(self.name), shape=(input_shape[-1],), initializer=self.init,
                                 regularizer=self.w_regularizer, constraint=self.w_constraint)
        self.features_dim = input_shape[-1]

        if self.bias:
            self.b = self.add_weight(name='{}_b'.format(self.name), shape=(input_shape[1],), initializer='zero',
                                     regularizer=self.b_regularizer, constraint=self.b_constraint)
        else:
            self.b = None

        self.build = True

    def compute_mask(self, inputs, mask=None):
        return None

    def call(self, x, mask=None, **kwargs):
        features_dim = self.features_dim
        step_dim = self.step_dim

        e = K.dot(K.reshape(x, (-1, features_dim)), K.reshape(self.w, (features_dim, 1)))
        e = K.reshape(e, (-1, step_dim))
        if self.bias:
            e += self.b
        e = K.tanh(e)

        a = K.exp(e)

        if mask is not None:
            a *= K.cast(mask, K.floatx())

        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        a = K.expand_dims(a)

        c = K.sum(a * x, axis=1)
        return c

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.features_dim
