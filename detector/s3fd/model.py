from keras.models import *
from keras.layers import *
import tensorflow as tf

class L2Norm(Layer):
    '''
    Code borrows from https://github.com/flyyufelix/cnn_finetune
    '''
    def __init__(self, weights=None, axis=-1, gamma_init='zero', n_channels=256, scale=10, **kwargs):
        self.axis = axis
        self.gamma_init = initializers.get(gamma_init)
        self.initial_weights = weights
        self.n_channels = n_channels
        self.scale = scale
        super(L2Norm, self).__init__(**kwargs)

    def build(self, input_shape):
        self.input_spec = [InputSpec(shape=input_shape)]
        self.gamma = K.variable(self.gamma_init((self.n_channels,)), name='{}_gamma'.format(self.name))
        self.trainable_weights = [self.gamma]
        self.built = True

    def call(self, x, mask=None):
        norm = K.sqrt(K.sum(K.square(x), axis=-1, keepdims=True)) + K.epsilon()
        x = x / norm * self.gamma
        return x

    def get_config(self):
        config = {"axis": self.axis}
        base_config = super(L2Norm, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

def s3fd_keras():    
    inp = Input((None,None,3))
    
    conv1_1 = Conv2D(filters=64, kernel_size=3, padding="same", name="conv1_1", activation="relu")(inp)
    conv1_2 = Conv2D(filters=64, kernel_size=3, padding="same", name="conv1_2", activation="relu")(conv1_1)
    maxpool1 = MaxPooling2D()(conv1_2)
    
    conv2_1 = Conv2D(filters=128, kernel_size=3, padding="same", name="conv2_1", activation="relu")(maxpool1)
    conv2_2 = Conv2D(filters=128, kernel_size=3, padding="same", name="conv2_2", activation="relu")(conv2_1)
    maxpool2 = MaxPooling2D()(conv2_2)
    
    conv3_1 = Conv2D(filters=256, kernel_size=3, padding="same", name="conv3_1", activation="relu")(maxpool2)
    conv3_2 = Conv2D(filters=256, kernel_size=3, padding="same", name="conv3_2", activation="relu")(conv3_1)
    conv3_3 = Conv2D(filters=256, kernel_size=3, padding="same", name="conv3_3", activation="relu")(conv3_2)
    f3_3 = conv3_3
    maxpool3 = MaxPooling2D()(conv3_3)
    
    conv4_1 = Conv2D(filters=512, kernel_size=3, padding="same", name="conv4_1", activation="relu")(maxpool3)
    conv4_2 = Conv2D(filters=512, kernel_size=3, padding="same", name="conv4_2", activation="relu")(conv4_1)
    conv4_3 = Conv2D(filters=512, kernel_size=3, padding="same", name="conv4_3", activation="relu")(conv4_2)
    f4_3 = conv4_3
    maxpool4 = MaxPooling2D()(conv4_3)
    
    conv5_1 = Conv2D(filters=512, kernel_size=3, padding="same", name="conv5_1", activation="relu")(maxpool4)
    conv5_2 = Conv2D(filters=512, kernel_size=3, padding="same", name="conv5_2", activation="relu")(conv5_1)
    conv5_3 = Conv2D(filters=512, kernel_size=3, padding="same", name="conv5_3", activation="relu")(conv5_2)
    f5_3 = conv5_3
    maxpool5 = MaxPooling2D()(conv5_3)
    
    
    # ========== Note ==========
    # Be careful about the zeropadding difference when strides >= 2
    fc6 = ZeroPadding2D(3)(maxpool5)
    fc6 = Conv2D(filters=1024, kernel_size=3, name="fc6", activation="relu")(fc6)
    fc7 = Conv2D(filters=1024, kernel_size=1, name="fc7", activation="relu")(fc6)
    ffc7 = fc7
    conv6_1 = Conv2D(filters=256, kernel_size=1, name="conv6_1", activation="relu")(fc7)
    f6_1 = conv6_1
    conv6_2 = ZeroPadding2D()(conv6_1)
    conv6_2 = Conv2D(filters=512, kernel_size=3, strides=2, name="conv6_2", activation="relu")(conv6_2)
    f6_2 = conv6_2
    conv7_1 = Conv2D(filters=128, kernel_size=1, name="conv7_1", activation="relu")(f6_2)
    f7_1 = conv7_1
    conv7_2 = ZeroPadding2D()(conv7_1)
    conv7_2 = Conv2D(filters=256, kernel_size=3, strides=2, name="conv7_2", activation="relu")(conv7_2)
    f7_2 = conv7_2
    
    f3_3 = L2Norm(n_channels=256, scale=10, name="conv3_3_norm")(f3_3)
    f4_3 = L2Norm(n_channels=512, scale=8, name="conv4_3_norm")(f4_3)
    f5_3 = L2Norm(n_channels=512, scale=5, name="conv5_3_norm")(f5_3)
    
    cls1 = Conv2D(filters=4, kernel_size=3, padding="same", name="conv3_3_norm_mbox_conf")(f3_3)
    reg1 = Conv2D(filters=4, kernel_size=3, padding="same", name="conv3_3_norm_mbox_loc")(f3_3)
    cls2 = Conv2D(filters=2, kernel_size=3, padding="same", name="conv4_3_norm_mbox_conf")(f4_3)
    reg2 = Conv2D(filters=4, kernel_size=3, padding="same", name="conv4_3_norm_mbox_loc")(f4_3)
    cls3 = Conv2D(filters=2, kernel_size=3, padding="same", name="conv5_3_norm_mbox_conf")(f5_3)
    reg3 = Conv2D(filters=4, kernel_size=3, padding="same", name="conv5_3_norm_mbox_loc")(f5_3)
    cls4 = Conv2D(filters=2, kernel_size=3, padding="same", name="fc7_mbox_conf")(ffc7)
    reg4 = Conv2D(filters=4, kernel_size=3, padding="same", name="fc7_mbox_loc")(ffc7)
    
    cls5 = Conv2D(filters=2, kernel_size=3, padding="same", name="conv6_2_mbox_conf")(f6_2)
    reg5 = Conv2D(filters=4, kernel_size=3, padding="same", name="conv6_2_mbox_loc")(f6_2)
    cls6 = Conv2D(filters=2, kernel_size=3, padding="same", name="conv7_2_mbox_conf")(f7_2)
    reg6 = Conv2D(filters=4, kernel_size=3, padding="same", name="conv7_2_mbox_loc")(f7_2)
    
    def get_chunk(x, c):
        return tf.split(x, c, axis=-1)
    chunk = Lambda(lambda x: get_chunk(x, 4))(cls1)
    bmax = Lambda(lambda chunk: K.maximum(K.maximum(chunk[0], chunk[1]), chunk[2]))(chunk)
    cls1 = Concatenate(axis=-1)([bmax, chunk[3]])
    
    return Model(inp, [cls1, reg1, cls2, reg2, cls3, reg3, cls4, reg4, cls5, reg5, cls6, reg6])