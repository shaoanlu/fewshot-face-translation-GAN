from keras.layers import *
from keras.layers.advanced_activations import LeakyReLU
import tensorflow as tf

from networks.instance_normalization import InstanceNormalization

conv_init = "he_normal"
w_l2 = 3e-5
norm = "instancenorm"

def ReflectPadding2D(x, pad=1):
    x = Lambda(lambda x: tf.pad(x, [[0, 0], [pad, pad], [pad, pad], [0, 0]], mode='REFLECT'))(x)
    return x

def normalization(inp, norm='none', group='16'):    
    x = inp
    if norm == 'batchnorm':
        x = BatchNormalization()(x)
    elif norm == 'instancenorm':
        x = InstanceNormalization()(x)
    #elif norm == "SPADE_norm":
    #    def spade_norm(x):
    #        meanC, varC = tf.nn.moments(x, [1, 2], keep_dims=True)
    #        sigmaC = tf.sqrt(tf.add(varC, 1e-5))
    #        return (content - meanC) / sigmaC
    #    x = Lambda(lambda xx: spade_norm(xx))(x)
    else:
        x = x
    return x

def conv_block(input_tensor, f, use_norm=False, k=3, strides=2):
    x = input_tensor
    if not k == 1:
        x = ReflectPadding2D(x)
    x = Conv2D(f, kernel_size=k, strides=strides, kernel_regularizer=regularizers.l2(w_l2),  
               kernel_initializer=conv_init, use_bias=(not use_norm))(x)
    x = normalization(x, norm, f) if use_norm else x
    x = Activation("relu")(x)
    return x

def conv_block_d(input_tensor, f, use_norm=False, k=3, strides=2):
    x = input_tensor
    if not k == 1:
        x = ReflectPadding2D(x)
    x = Conv2D(f, kernel_size=k, strides=strides, kernel_regularizer=regularizers.l2(w_l2), 
               kernel_initializer=conv_init, use_bias=(not use_norm))(x)
    x = normalization(x, 'instancenorm', f) if use_norm else x
    x = LeakyReLU(alpha=0.2)(x)   
    return x

def res_block(input_tensor, f, use_norm=True):
    x = input_tensor
    
    x = ReflectPadding2D(x)
    x = Conv2D(f, kernel_size=3, kernel_regularizer=regularizers.l2(w_l2), 
               kernel_initializer=conv_init, use_bias=not use_norm)(x)
    x = normalization(x, norm, f) if use_norm else x
    x = Activation('relu')(x)
    x = ReflectPadding2D(x)
    x = Conv2D(f, kernel_size=3, kernel_regularizer=regularizers.l2(w_l2), 
               kernel_initializer=conv_init)(x)
    x = add([x, input_tensor])
    x = Activation('relu')(x)
    return x

def embddding_fc_block(input_tensor):
    x = input_tensor
    
    x = Dense(256, kernel_regularizer=regularizers.l2(w_l2))(x)
    x = normalization(x, norm, 256)
    x = Activation('relu')(x)
    x = Dense(256, kernel_regularizer=regularizers.l2(w_l2))(x)
    x = normalization(x, norm, 256)
    x = Activation('relu')(x)
    x = Dense(256, kernel_regularizer=regularizers.l2(w_l2))(x)
    x = normalization(x, norm, 256)
    x = Activation('relu')(x)
    return x

def adain_resblock(input_tensor, embeddings, f):
    # conv -> norm -> activ -> conv -> norm -> add
    def AdaIN(content, style_var, style_mean, epsilon=1e-5):
        meanC, varC = tf.nn.moments(content, [1, 2], keep_dims=True)
        sigmaC = tf.sqrt(tf.add(varC, epsilon))
        return (content - meanC) * style_var / sigmaC + style_mean
    
    x = input_tensor
    style_var = Conv2D(f, 1, strides=1)(embeddings)
    style_mean = Conv2D(f, 1, strides=1)(embeddings)
    
    x = ReflectPadding2D(x)
    x = Conv2D(f, kernel_size=3, kernel_regularizer=regularizers.l2(w_l2), 
               kernel_initializer=conv_init, use_bias=False)(x)
    x = Lambda(lambda x: AdaIN(x[0], x[1], x[2]))([x, style_var, style_mean])
    x = Activation('relu')(x)
    x = ReflectPadding2D(x)
    x = Conv2D(f, kernel_size=3, kernel_regularizer=regularizers.l2(w_l2), 
               kernel_initializer=conv_init, use_bias=False)(x)
    x = Lambda(lambda x: AdaIN(x[0], x[1], x[2]))([x, style_var, style_mean])
    x = add([x, input_tensor])
    x = Activation('relu')(x)
    return x

def SPADE(input_tensor, cond_input_tensor, f, block_id=0):
    x = input_tensor
    x = InstanceNormalization(name=f"SPADE_norm{block_id}")(x)# x = normalization(x, "SPADE_norm", f)
    y = cond_input_tensor
    y = Conv2D(128, kernel_size=3, padding='same')(y)
    y = Activation('relu')(y)           
    gamma = Conv2D(f, kernel_size=3, padding='same')(y)
    beta = Conv2D(f, kernel_size=3, padding='same')(y)    
    x = add([x, multiply([x, gamma])])
    x = add([x, beta])
    return x

def SPADE_res_block(input_tensor, cond_input_tensor, f, block_id=0):
    x = input_tensor
    x = SPADE(x, cond_input_tensor, f, block_id=block_id+"_0")
    x = Activation('relu')(x)
    x = Conv2D(f, kernel_size=3, kernel_initializer=conv_init, padding='same')(x)
    x = SPADE(x, cond_input_tensor, f, block_id=block_id+"_1")
    x = Activation('relu')(x)
    x = Conv2D(f, kernel_size=3, kernel_initializer=conv_init, padding='same')(x)
    x = add([x, input_tensor])
    x = Activation('relu')(x)
    return x

def upscale_nn(input_tensor, f, use_norm=True, w_l2=w_l2, norm=norm):
    x = input_tensor
    x = UpSampling2D()(x)
    x = Conv2D(f, kernel_size=4, kernel_regularizer=regularizers.l2(w_l2), 
               kernel_initializer=conv_init, padding='same')(x)
    x = normalization(x, norm, f) if use_norm else x
    x = LeakyReLU(0.2)(x)
    return x

def resize_tensor(inp, shape):
    if isinstance(shape, int):
        return Lambda(lambda x: tf.image.resize_images(x, [shape, shape]))(inp)
    elif isinstance(shape, list):
        return Lambda(lambda x: tf.image.resize_images(x, [shape[0], shape[1]]))(inp)
