# https://github.com/philipperemy/keras-attention-mechanism/blob/master/attention/attention.py

from keras import layers


class SimpleAttention(layers.Layer):
    def __init__(self, **kwargs):
        super(SimpleAttention, self).__init__(**kwargs)

    def __call__(self, hidden_states, *args, **kwargs):
        """
        Many-to-one attention mechanism for Keras.
        @param hidden_states: 3D tensor with shape (batch_size, time_steps, input_dim).
        @return: 2D tensor with shape (batch_size, 128)
        """
        hidden_size = int(hidden_states.shape[2])
        # Inside dense layer
        #              hidden_states            dot               W            =>           score_first_part
        # (batch_size, time_steps, hidden_size) dot (hidden_size, hidden_size) => (batch_size, time_steps, hidden_size)
        # W is the trainable weight matrix of attention Luong's multiplicative style score
        score_first_part = layers.Dense(hidden_size, use_bias=False, name='attention_score_vec')(hidden_states)
        #            score_first_part           dot        last_hidden_state     => attention_weights
        # (batch_size, time_steps, hidden_size) dot   (batch_size, hidden_size)  => (batch_size, time_steps)
        h_t = layers.Lambda(lambda x: x[:, -1, :], output_shape=(hidden_size,), name='last_hidden_states')(
            hidden_states)
        score = layers.dot(inputs=[score_first_part, h_t], axes=[2, 1], name='attention_score')
        attention_weights = layers.Activation(activation='softmax', name='attention_weight')(score)
        # (batch_size, time_steps, hidden_size) dot (batch_size, time_steps) => (batch_size, hidden_size)
        content_vector = layers.dot(inputs=[hidden_states, attention_weights], axes=[1, 1], name='content_vector')
        pre_activation = layers.concatenate([content_vector, h_t], name='attention_output')
        attention_vector = layers.Dense(hidden_size // 2, use_bias=False, activation='tanh', name='attention_vector')(
            pre_activation)
        return attention_vector


if __name__ == '__main__':
    from deep_learning.knowledge_graph.relation_extract.bilstm_attention_keras import BiLSTMAttention

    b = BiLSTMAttention('', '')
    b.predict()
