import os
import time

import numpy as np
import keras.backend as K
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras import layers, models, optimizers, constraints, utils

from deep_learning.elmo.custom_layers import TimeStepDropout, HighWay, Camouflage, SampleSoftmax


class ELMo:
    def __init__(self, parameters):
        self._model = None
        self._elmo_model = None
        self.parameters = parameters

    def __del__(self):
        K.clear_session()
        del self._model

    def char_level_token_encoder(self):
        charset_size = self.parameters.get('charset_size')
        char_embedding_size = self.parameters.get('char_embedding_size')
        token_embedding_size = self.parameters.get('hidden_units_size')
        n_highway_layers = self.parameters.get('n_highway_layers')
        filters = self.parameters.get('cnn_filters')
        token_maxlen = self.parameters.get('token_maxlen')

        # Input Layer, word characters (samples, words, character_indices)
        inputs = layers.Input(shape=(None, token_maxlen,), dtype='int32')
        # Embed characters (samples, words, characters, character embedding)
        embedding = layers.Embedding(input_dim=charset_size, output_dim=char_embedding_size)(inputs)

        token_embeds = []
        # Apply multi-filter 2D convolutions + 1D MaxPooling + tanh
        for (window_size, filters_size) in filters:
            convs = layers.Conv2D(filters=filters_size, kernel_size=(window_size, char_embedding_size), strides=(1, 1),
                                  padding='same')(embedding)
            convs = layers.TimeDistributed(layers.GlobalMaxPool1D())(convs)
            convs = layers.Activation(activation='tanh')(convs)
            convs = Camouflage(mask_value=0)(inputs=[convs, inputs])
            token_embeds.append(convs)
        token_embeds = layers.concatenate(token_embeds)

        # Apply highways networks
        for i in range(n_highway_layers):
            token_embeds = layers.TimeDistributed(HighWay())(token_embeds)
            token_embeds = Camouflage(mask_value=0)(input=[token_embeds, inputs])

        # Project to token embedding dimensionality
        token_embeds = layers.TimeDistributed(layers.Dense(units=token_embedding_size, activation='linear'))(
            token_embeds)
        token_embeds = Camouflage(mask_value=0)(input=[token_embeds, inputs])

        token_encoder = models.Model(inputs=inputs, outputs=token_embeds, name='token_encoding')
        return token_encoder

    def compile_elmo(self):
        """
        Compiles a Language Model RNN based on the given parameters
        """
        if self.parameters.get('token_encoding') == 'word':
            # Train word embeddings from scratch
            word_inputs = layers.Input(shape=(None,), name='word_indices', dtype='int32')
            embedding = layers.Embedding(input_dim=self.parameters.get('vocab_size'),
                                         output_dim=self.parameters.get('hidden_units_size'), trainable=True,
                                         name='token_encoding')
            inputs = embedding(word_inputs)

            # Token embeddings for Input
            drop_inputs = layers.SpatialDropout1D(self.parameters.get('dropout_rate'))(inputs)
            lstm_inputs = TimeStepDropout(self.parameters.get('word_dropout_rate'))(drop_inputs)

            # Pass outputs as inputs to apply sampled softmax
            next_ids = layers.Input(shape=(None, 1), name='next_ids', dtype='float32')
            previous_ids = layers.Input(shape=(None, 1), name='previous_ids', dtype='float32')
        elif self.parameters.get('token_encoding') == 'char':
            # Train character-level representation
            word_inputs = layers.Input(shape=(None, self.parameters.get('token_maxlen')), dtype='int32',
                                       name='char_indices')
            inputs = self.char_level_token_encoder()(word_inputs)

            # Token embeddings for Input
            drop_inputs = layers.SpatialDropout1D(self.parameters.get('dropout_rate'))(inputs)
            lstm_inputs = TimeStepDropout(self.parameters.get('word_dropout_rate'))(inputs)

            # Pass outputs as inputs to apply sampled softmax
            next_ids = layers.Input(shape=(None, 1), name='next_ids', dtype='float32')
            previous_ids = layers.Input(shape=(None, 1), name='previous_ids', dtype='float32')

        # Reversed input for backward LSTMs
        re_lstm_inputs = layers.Lambda(function=ELMo.reverse)(lstm_inputs)
        mask = layers.Lambda(function=ELMo.reverse)(drop_inputs)

        # Forward LSTMs
        for i in range(self.parameters.get('n_lstm_layers')):
            if self.parameters['cuDNN']:
                lstm = layers.CuDNNLSTM(units=self.parameters.get('lstm_units_size'), return_sequences=True,
                                        kernel_constraint=constraints.MinMaxNorm(-1 * self.parameters.get('cell_clip'),
                                                                                 self.parameters('cell_clip')),
                                        recurrent_constraint=constraints.MinMaxNorm(
                                            -1 * self.parameters.get('cell_clip'),
                                            self.parameters.get('cell_clip'))
                                        )(lstm_inputs)
            else:
                lstm = layers.LSTM(units=self.parameters.get('lstm_units_size'), return_sequences=True,
                                   activation='tanh', recurrent_activation='sigmoid',
                                   kernel_constraint=constraints.MinMaxNorm(-1 * self.parameters.get('cell_clip'),
                                                                            self.parameters.get('cell_clip')),
                                   recurrent_constraint=constraints.MinMaxNorm(-1 * self.parameters.get('cell_clip'),
                                                                               self.parameters.get('cell_clip')))(
                    lstm_inputs)

            lstm = Camouflage(mask_value=0)(inputs=[lstm, drop_inputs])

            # Projection to hidden_units_size
            proj = layers.TimeDistributed(layers.Dense(self.parameters.get('hidden_units_size'), activation='linear',
                                                       kernel_constraint=constraints.MinMaxNorm(
                                                           -1 * self.parameters.get('proj_clip'),
                                                           self.parameters.get('proj_clip'))))(lstm)

            # Merge Bi-LSTMs feature vectors with the previous ones
            lstm_inputs = layers.add([proj, lstm_inputs], name='f_block_{}'.format(i + 1))
            # Apply variational drop-out between BI-LSTM layers
            lstm_inputs = layers.SpatialDropout1D(self.parameters.get('dropout_rate'))(lstm_inputs)

        # Backward LSTMs
        for i in range(self.parameters.get('n_lstm_layers')):
            if self.parameters['cuDNN']:
                re_lstm = layers.CuDNNLSTM(units=self.parameters.get('lstm_units_size'), return_sequences=True,
                                           kernel_constraint=constraints.MinMaxNorm(
                                               -1 * self.parameters.get('cell_clip'), self.parameters.get('cell_clip')),
                                           recurrent_constraint=constraints.MinMaxNorm(
                                               -1 * self.parameters('cell_clip'), self.parameters.get('cell_clip')))(
                    re_lstm_inputs)
            else:
                re_lstm = layers.LSTM(units=self.parameters.get('lstm_units_size'), return_sequences=True,
                                      activation='tanh', recurrent_activation='sigmoid',
                                      kernel_constraint=constraints.MinMaxNorm(-1 * self.parameters.get('cell_clip'),
                                                                               self.parameters.get('cell_clip')),
                                      recurrent_constraint=constraints.MinMaxNorm(-1 * self.parameters.get('cell_clip'),
                                                                                  self.parameters.get('cell_clip')))(
                    re_lstm_inputs)
            re_lstm = Camouflage(mask_value=0)(inputs=[re_lstm, mask])
            # Projection to hidden_units_size
            re_proj = layers.TimeDistributed(layers.Dense(self.parameters.get('hidden_units_size'), activation='linear',
                                                          kernel_constraint=constraints.MinMaxNorm(
                                                              -1 * self.parameters.get('proj_clip'),
                                                              self.parameters.get('proj_clip'))))(re_lstm)
            # Merge Bi-LSTMs feature vectors with the previous ones
            re_lstm_inputs = layers.add([re_proj, re_lstm_inputs], name='b_block_{}'.format(i + 1))
            # Apply variational drop-out between BI-LSTM layers
            re_lstm_inputs = layers.SpatialDropout1D(self.parameters.get('dropout_rate'))(re_lstm_inputs)

        # Reverse backward LSTMs' outputs = Make it forward again
        re_lstm_inputs = layers.Lambda(function=ELMo.reverse, name='reverse')(re_lstm_inputs)

        # Project to Vocabulary with Sampled Softmax
        sampled_softmax = SampleSoftmax(num_classes=self.parameters.get('vocab_size'),
                                        num_sampled=int(self.parameters.get('num_sampled')),
                                        tied_to=embedding if self.parameters.get(
                                            'weight_tying') and self.parameters.get(
                                            'token_encoding') == 'word' else None)
        outputs = sampled_softmax([lstm_inputs, next_ids])
        re_outputs = sampled_softmax([re_lstm_inputs, previous_ids])

        self._model = models.Model(inputs=[word_inputs, next_ids, previous_ids], outputs=[outputs, re_outputs])
        self._model.compile(
            optimizer=optimizers.Adagrad(lr=self.parameters.get('lr'), clipvalue=self.parameters.get('clip_value')),
            loss=None)
        print(self._model.summary())

    def train(self, train_data, valid_data):
        # Add callbacks (early stopping, model checkpoint)
        weights_file = os.path.join('')
        save_best_model = ModelCheckpoint(filepath=weights_file, monitor='val_loss', verbose=1,
                                          save_best_only=True, mode='auto')
        early_stop = EarlyStopping(patience=self.parameters.get('patience'), restore_best_weights=True)

        self._model.fit(train_data, validation_data=valid_data, epochs=self.parameters.get('epochs'),
                        callbacks=[save_best_model, early_stop])

    @staticmethod
    def reverse(inputs, axes=-1):
        return K.reverse(inputs, axes=axes)

    def evaluate(self, test_data):
        def unpad(x, y_true, y_pred):
            y_true_unpad = []
            y_pred_unpad = []
            for i, x_i in enumerate(x):
                for j, x_ij in enumerate(x_i):
                    if x_ij == 0:
                        y_true_unpad.append(y_true[i][:j])
                        y_pred_unpad.append(y_pred[i][:j])
                        break
            return np.asarray(y_true_unpad), np.asarray(y_pred_unpad)
