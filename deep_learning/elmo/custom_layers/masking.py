import keras.backend as K
from keras.layers import Layer


class Camouflage(Layer):
    """Masks a sequence by using a mask value to skip timesteps based on another sequence.
       LSTM and Convolution layers may produce fake tensors for padding timesteps. We need
       to eliminate those tensors by replicating their initial values presented in the second input.
       inputs = Input()
       lstms = LSTM(units=100, return_sequences=True)(inputs)
       padded_lstms = Camouflage()([lstms, inputs])
       ...
    """

    def __init__(self, mask_value=0, **kwargs):
        super(Camouflage, self).__init__(**kwargs)
        self.mask_value = mask_value

    def call(self, inputs, **kwargs):
        boolean_mask = K.any(K.not_equal(inputs[1], self.mask_value), axis=-1, keepdims=True)
        return inputs[0] * K.cast(boolean_mask, K.dtype(inputs[0]))

    def get_config(self):
        config = {
            'mask_value': self.mask_value
        }
        base_config = super(Camouflage, self).get_config()
        return base_config.update(config)

    def compute_output_shape(self, input_shape):
        return input_shape[0]
