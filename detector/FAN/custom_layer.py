from keras.layers import Layer
from keras import backend as K

class TorchBatchNorm2D(Layer):
        def __init__(self, axis=-1, momentum=0.99, epsilon=1e-3, **kwargs):
            super(TorchBatchNorm2D, self).__init__(**kwargs)
            self.axis = axis
            self.momentum = momentum
            self.epsilon = epsilon

        def build(self, input_shape):
            dim = input_shape[self.axis]
            if dim is None:
                raise ValueError('Axis ' + str(self.axis) + ' of ' 'input tensor should have a defined dimension ' 'but the layer received an input with shape ' + str(input_shape) + '.')
            shape = (dim,)
            self.gamma = self.add_weight(shape=shape, name='gamma', initializer='ones', regularizer=None, constraint=None)
            self.beta = self.add_weight(shape=shape, name='beta', initializer='zeros', regularizer=None, constraint=None)
            self.moving_mean = self.add_weight(shape=shape, name='moving_mean', initializer='zeros', trainable=False)            
            self.moving_variance = self.add_weight(shape=shape, name='moving_variance', initializer='ones', trainable=False)            
            self.built = True

        def call(self, inputs, training=None):
            # this custom batchnorm does not support training phase update
            input_shape = K.int_shape(inputs)

            broadcast_shape = [1] * len(input_shape)
            broadcast_shape[self.axis] = input_shape[self.axis]

            broadcast_moving_mean = K.reshape(self.moving_mean, broadcast_shape)
            broadcast_moving_variance = K.reshape(self.moving_variance, broadcast_shape)
            broadcast_gamma = K.reshape(self.gamma, broadcast_shape)
            broadcast_beta = K.reshape(self.beta, broadcast_shape)
        
            def normalize_inference():
                return K.batch_normalization(
                    inputs,
                    broadcast_moving_mean,
                    broadcast_moving_variance,
                    broadcast_beta,
                    broadcast_gamma,
                    #axis=self.axis,
                    epsilon=self.epsilon)
            
            return normalize_inference()            

        def get_config(self):
            config = { 'axis': self.axis, 'momentum': self.momentum, 'epsilon': self.epsilon }
            base_config = super(TorchBatchNorm2D, self).get_config()
            return dict(list(base_config.items()) + list(config.items()))