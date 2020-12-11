# https://github.com/bojone/seq2seq/blob/master/seq2seq.py
import os
import json
import tensorflow as tf
import numpy as np
from tqdm import tqdm
import pickle
import keras
import joblib
from keras.layers import *
from keras_layer_normalization import LayerNormalization
from keras.models import Model, load_model
from keras.callbacks import Callback
from keras.optimizers import Adam
from keras.utils import get_custom_objects

from corpus.load_corpus import LoadCorpus
from corpus import seq2seq_config_path, seq2seq_data_path, corpus_root_path

min_count = 32
maxlen = 50
batch_size = 32
char_size = 128
z_dim = 128
epochs = 100

if os.path.exists(seq2seq_config_path):
    chars, id2char, char2id = json.load(open(seq2seq_config_path, mode='r'))
    id2char = dict((int(id), char) for id, char in id2char.items())
else:
    chars = {}
    x, y = LoadCorpus.load_xiaohuangji_train()
    for i, j in tqdm(zip(x, y)):
        for ic in i:
            chars[ic] = chars.get(ic, 0) + 1
        for jc in j:
            chars[jc] = chars.get(jc, 0) + 1
    chars = dict((i, j) for i, j in chars.items() if j >= min_count)
    # 0 mask  1 unk  2 start  3 end
    id2char = dict((i + 4, j) for i, j in enumerate(chars))
    char2id = dict((j, i) for i, j in id2char.items())
    json.dump([chars, id2char, char2id], open(seq2seq_config_path, mode='w'))


def str2id(s, start_end=False):
    if start_end:
        ids = [2] + [char2id.get(i, 1) for i in s[:maxlen - 2]] + [3]
    else:
        ids = [char2id.get(i, 1) for i in s[:maxlen]]
    return ids


def id2str(ids):
    return ''.join([id2char.get(i, '') for i in ids])


def padding(x):
    ml = max([len(i) for i in x])
    return [i + [0] * (ml - len(i)) for i in x]


def data_generator():
    if os.path.exists(seq2seq_data_path):
        with open(seq2seq_data_path, mode='rb') as f:
            x = pickle.load(f)
            y = pickle.load(f)
    else:
        x, y = LoadCorpus.load_xiaohuangji_train()
        x = [i.replace(' ', '') for i in x]
        y = [i.replace(' ', '') for i in y]
        x = np.array(padding([str2id(i) for i in x]))
        y = np.array(padding([str2id(i, start_end=True) for i in y]))
        with open(seq2seq_data_path, mode='wb') as f:
            pickle.dump(x, f)
            pickle.dump(y, f)
    # x, y = x[:1000], y[:1000]
    return [x, y], None


def to_one_hot(x):
    x, x_mask = x
    x = K.cast(x, 'int32')
    x = K.one_hot(x, len(chars) + 4)
    x = K.sum(x_mask * x, 1, keepdims=True)
    x = K.cast(K.greater(x, 0.5), 'float32')
    return x


class ScaleShift(Layer):
    def __init__(self, **kwargs):
        super(ScaleShift, self).__init__(**kwargs)

    def build(self, input_shape):
        kernel_shape = (1,) * (len(input_shape) - 1) + (input_shape[-1],)
        self.log_scale = self.add_weight(name='log_scale', shape=kernel_shape, initializer='zeros')
        self.shift = self.add_weight(name='shift', shape=kernel_shape, initializer='zeros')

    def call(self, inputs, **kwargs):
        x_outs = K.exp(self.log_scale) * inputs + self.shift
        return x_outs


class OurLayer(Layer):
    def reuse(self, layer, *args, **kwargs):
        if not layer.built:
            if len(args) > 0:
                inputs = args[0]
            else:
                inputs = kwargs['inputs']
            if isinstance(inputs, list):
                input_shape = [K.int_shape(x) for x in inputs]
            else:
                input_shape = K.int_shape(inputs)
            layer.build(input_shape)
        outputs = layer.call(*args, **kwargs)
        if not keras.__version__.startswith('2.3.'):
            for w in layer.trainable_weights:
                if w not in self._trainable_weights:
                    self._trainable_weights.append(w)
            for w in layer.non_trainable_weights:
                if w not in self.non_trainable_weights:
                    self.non_trainable_weights.append(w)
            for u in layer.updates:
                if not hasattr(self, '_updates'):
                    self._updates = []
                if u not in self._updates:
                    self._updates.append(u)
        return outputs


class OurBidirectional(OurLayer):
    def __init__(self, layer, **kwargs):
        super(OurBidirectional, self).__init__(**kwargs)
        self.forward_layer = layer.__class__.from_config(layer.get_config())
        self.backward_layer = layer.__class__.from_config(layer.get_config())
        self.forward_layer.name = 'forward_' + self.forward_layer.name
        self.backward_layer.name = 'backward_' + self.backward_layer.name

    def reverse_sequence(self, x, mask):
        seq_len = K.round(K.sum(mask, 1)[:, 0])
        seq_len = K.cast(seq_len, 'int32')
        return K.tf.reverse_sequence(x, seq_len, seq_dim=1)

    def call(self, inputs, **kwargs):
        x, mask = inputs
        x_forward = self.reuse(self.forward_layer, x)
        x_backward = self.reverse_sequence(x, mask)
        x_backward = self.reuse(self.backward_layer, x_backward)
        x_backward = self.reverse_sequence(x_backward, mask)
        x = K.concatenate([x_forward, x_backward], axis=-1)
        if K.ndim(x) == 3:
            return x * mask
        else:
            return x

    def compute_output_shape(self, input_shape):
        return input_shape[0][:-1] + (self.forward_layer.units * 2,)

    def get_config(self):
        base_config = super(OurBidirectional, self).get_config()
        print(self.forward_layer.get_config())
        config = {
            'forward_layer': self.forward_layer.get_config(),
            'backward_layer': self.backward_layer.get_config(),
            # 'layer': self.layer.get_config()
        }
        return dict(list(base_config.items()) + list(config.items()))


def seq_avgpool(x):
    seq, mask = x
    return K.sum(seq * mask, 1) / (K.sum(mask, 1) + 1e-6)


def seq_maxpool(x):
    seq, mask = x
    seq -= (1 - mask) * 1e10
    return K.max(seq, 1)


class SelfModulatedLayerNormalization(OurLayer):
    def __init__(self, num_hidden, **kwargs):
        super(SelfModulatedLayerNormalization, self).__init__(**kwargs)
        self.num_hidden = num_hidden

    def build(self, input_shape):
        super(SelfModulatedLayerNormalization, self).build(input_shape)
        output_dim = input_shape[0][-1]
        self.layernorm = LayerNormalization(center=False, scale=False)
        self.beta_dense_1 = Dense(self.num_hidden, activation='relu')
        self.beta_dense_2 = Dense(output_dim)
        self.gamma_dense_1 = Dense(self.num_hidden, activation='relu')
        self.gamma_dense_2 = Dense(output_dim)

    def call(self, inputs, **kwargs):
        inputs, cond = inputs
        inputs = self.reuse(self.layernorm, inputs)
        beta = self.reuse(self.beta_dense_1, cond)
        beta = self.reuse(self.beta_dense_2, beta)
        gamma = self.reuse(self.gamma_dense_1, cond)
        gamma = self.reuse(self.gamma_dense_2, gamma)
        for _ in range(K.ndim(inputs) - K.ndim(cond)):
            beta = K.expand_dims(beta, 1)
            gamma = K.expand_dims(gamma, 1)
        return inputs * (gamma + 1) + beta

    def compute_output_shape(self, input_shape):
        return input_shape[0]

    def get_config(self):
        base_config = super(SelfModulatedLayerNormalization, self).get_config()
        config = {'num_hidden': self.num_hidden}
        return dict(list(base_config.items()) + list(config.items()))


class Attention(OurLayer):
    def __init__(self, heads, size_per_head, key_size=None, mask_right=False, **kwargs):
        super(Attention, self).__init__(**kwargs)
        self.heads = heads
        self.size_per_head = size_per_head
        self.out_dim = heads * size_per_head
        self.key_size = key_size if key_size else size_per_head
        self.mask_right = mask_right

    def build(self, input_shape):
        super(Attention, self).build(input_shape)
        self.q_dense = Dense(self.key_size * self.heads, use_bias=False)
        self.k_dense = Dense(self.key_size * self.heads, use_bias=False)
        self.v_dense = Dense(self.out_dim, use_bias=False)

    def mask(self, x, mask, mode='mul'):
        if mask is None:
            return x
        else:
            for _ in range(K.ndim(x) - K.ndim(mask)):
                mask = K.expand_dims(mask, K.ndim(mask))
            if mode == 'mul':
                return x * mask
            else:
                return x - (1 - mask) * 1e10

    def call(self, inputs, **kwargs):
        q, k, v = inputs[:3]
        v_mask, q_mask = None, None
        if len(inputs) > 3:
            v_mask = inputs[3]
            if len(inputs) > 4:
                q_mask = inputs[4]

        # 线性变换
        qw = self.reuse(self.q_dense, q)
        kw = self.reuse(self.k_dense, k)
        vw = self.reuse(self.v_dense, v)

        # 形状变换
        qw = K.reshape(qw, (-1, K.shape(qw)[1], self.heads, self.key_size))
        kw = K.reshape(kw, (-1, K.shape(kw)[1], self.heads, self.key_size))
        vw = K.reshape(vw, (-1, K.shape(vw)[1], self.heads, self.size_per_head))

        # 维度置换
        qw = K.permute_dimensions(qw, (0, 2, 1, 3))
        kw = K.permute_dimensions(kw, (0, 2, 1, 3))
        vw = K.permute_dimensions(vw, (0, 2, 1, 3))

        # Attention
        a = K.tf.einsum('ijkl,ijml->ijkm', qw, kw) / self.key_size ** 0.5
        a = K.permute_dimensions(a, (0, 3, 2, 1))
        a = self.mask(a, v_mask, 'add')
        a = K.permute_dimensions(a, (0, 3, 2, 1))
        if self.mask_right:
            ones = K.ones_like(a[:1, :1])
            mask = (ones - K.tf.matrix_band_part(ones, -1, 0)) * 1e10
            a = a - mask

        a = K.softmax(a)

        # 完成输出
        o = K.tf.einsum('ijkl,ijlm->ijkm', a, vw)
        o = K.permute_dimensions(o, (0, 2, 1, 3))
        o = K.reshape(o, (-1, K.shape(o)[1], self.out_dim))
        o = self.mask(o, q_mask, 'mul')
        return o

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], input_shape[0][1], self.out_dim)


def build_model():
    x_in = Input(shape=(None,))
    y_in = Input(shape=(None,))
    x, y = x_in, y_in

    x_mask = Lambda(lambda x: K.cast(K.greater(K.expand_dims(x, 2), 0), 'float32'))(x)
    y_mask = Lambda(lambda x: K.cast(K.greater(K.expand_dims(x, 2), 0), 'float32'))(y)

    # x_one_hot = Lambda(to_one_hot)([x, x_mask])
    # x_prior = ScaleShift()(x_one_hot)  # 学习输出的先验分布，标题可能出现在文章中

    embedding = Embedding(len(chars) + 4, char_size)
    x = embedding(x)
    y = embedding(y)

    # encoder 双向双层LSTM
    x = LayerNormalization()(x)
    x = OurBidirectional(LSTM(z_dim // 2, return_sequences=True))([x, x_mask])
    x = LayerNormalization()(x)
    x = OurBidirectional(LSTM(z_dim // 2, return_sequences=True))([x, x_mask])
    x_max = Lambda(seq_maxpool)([x, x_mask])

    # deocer 单向双层lstm
    y = SelfModulatedLayerNormalization(z_dim // 4)([y, x_max])
    y = LSTM(z_dim, return_sequences=True)(y)
    y = SelfModulatedLayerNormalization(z_dim // 4)([y, x_max])
    y = LSTM(z_dim, return_sequences=True)(y)
    y = SelfModulatedLayerNormalization(z_dim // 4)([y, x_max])

    # attention
    xy = Attention(8, 16)([y, x, x, x_mask])
    xy = Concatenate()([y, xy])

    # 输出分类
    xy = Dense(char_size)(xy)
    xy = LeakyReLU(0.2)(xy)
    xy = Dropout(0.2)(xy)
    xy = Dense(len(chars) + 4)(xy)
    # xy = Lambda(lambda x: (x[0] + x[1]) / 2)([xy, x_prior])
    xy = Lambda(lambda x: (x[0]) / 2)([xy])
    xy = Activation('softmax')(xy)

    # 交叉熵损失，mask掉padding部分
    cross_entropy = K.sparse_categorical_crossentropy(y_in[:, 1:], xy[:, :-1])
    cross_entropy = K.sum(cross_entropy * y_mask[:, 1:, 0]) / K.sum(y_mask[:, 1:, 0])

    model = Model([x_in, y_in], xy)
    model.add_loss(cross_entropy)
    model.compile(optimizer=Adam(1e-3), metrics=['accuracy'])
    print(model.summary())
    return model


def gen_sent(model, s, topk=3, maxlen=64):
    # beam search
    xid = np.array([str2id(s)] * topk)
    yid = np.array([[2]] * topk)
    scores = [0] * topk
    for i in range(maxlen):
        proba = model.predict([xid, yid])[:, i, 3:]
        log_proba = np.log(proba + 1e-6)
        arg_topk = log_proba.argsort(axis=1)[:, -topk:]
        _yid = []
        _scores = []
        if i == 0:
            for j in range(topk):
                _yid.append(list(yid[j]) + [arg_topk[0][j] + 3])
                _scores.append(scores[j] + log_proba[0][arg_topk[0][j]])
        else:
            for i in range(topk):
                for k in range(topk):
                    _yid.append(list(yid[j]) + [arg_topk[j][k] + 3])
                    _scores.append(scores[j] + log_proba[j][arg_topk[j][k]])
            _arg_topk = np.argsort(_scores)[-topk:]
            _yid = [_yid[k] for k in _arg_topk]
        yid = np.array(_yid)
        scores = np.array(_scores)

        best_one = np.argmax(scores)
        if yid[best_one][-1] == 3:
            return id2str(yid[best_one])
    return id2str(yid[np.argmax(scores)])


class Evaluate(Callback):
    def __init__(self):
        super(Evaluate, self).__init__()
        self.lowest = 1e10
        self.weights_name = 'seq2seq.weights.2epoch_[{}]_val_loss_[{:.4f}]_loss_[{:.4f}]'

    def on_epoch_end(self, epoch, logs=None):
        global model
        # 保存最优结果
        if logs['loss'] <= self.lowest:
            self.lowest = logs['loss']
            model.save_weights(os.path.join(corpus_root_path, 'seq2seq',
                                            self.weights_name.format(epoch, logs['val_loss'], self.lowest)))


def running():
    x, y = data_generator()
    model.load_weights(os.path.join(corpus_root_path, 'seq2seq', 'seq2seq.weights'))
    model.fit(x, y, epochs=25, validation_split=0.1, shuffle=True, batch_size=batch_size, callbacks=[Evaluate()])
    model.save_weights(os.path.join(corpus_root_path, 'seq2seq', 'seq2seq.weights'))


def predict_sent():
    model.load_weights(os.path.join(corpus_root_path, 'seq2seq', 'seq2seq.weights'))
    s = ''
    d = gen_sent(model, s)
    print(d)


get_custom_objects().update(
    {'LayerNormalization': LayerNormalization, 'OurBidirectional': OurBidirectional, 'OurLayer': OurLayer})

model = build_model()

if __name__ == '__main__':
    running()
    # predict_sent()
