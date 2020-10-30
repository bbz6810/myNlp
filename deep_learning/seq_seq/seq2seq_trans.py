import math
import random
from keras import layers, models
import keras.backend as K


class Encoder(layers.Layer):
    def __init__(self, input_size, embed_size, hidden_size, n_layer=1, dropout=0.3, **kwargs):
        super(Encoder, self).__init__(**kwargs)
        self.input_size = input_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.n_layer = n_layer
        self.dropout = dropout

    def build(self, input_shape):
        super(Encoder, self).build(input_shape)
        self.embed = layers.Embedding(input_dim=self.input_size, output_dim=self.embed_size)
        self.gru = layers.Bidirectional(layers.GRU(units=self.hidden_size, dropout=self.dropout, name='gru'))

    def call(self, inputs, **kwargs):
        embedded = self.embed(inputs)
        outputs, hidden = self.gru(embedded)
        outputs = (outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:])
        return outputs


class Attention(layers.Layer):
    def __init__(self, hidden_size, **kwargs):
        super(Attention, self).__init__(**kwargs)
        self.hidden_size = hidden_size

    def build(self, input_shape):
        self.attn = layers.Dense(self.hidden_size)
        self.v = self.add_weight(shape=(self.hidden_size,), initializer='ones', name='attention_v')
        stdv = 1.0 / math.sqrt(K.eval(K.shape(self.v))[0])
        self.v = K.random_uniform(shape=K.shape(self.v), minval=-stdv, maxval=stdv)
        print(self.v)

    def call(self, inputs, **kwargs):
        # hidden, encoder_outputs = inputs[:2]
        # timestep = encoder_outputs.shape[0]
        # h = K.repeat(hidden, (timestep, 1, 1))
        h = K.repeat(K.ones(shape=(2,2)), 2)
        print(h)

if __name__ == '__main__':
    att = Attention(10)
    att.call('')
