from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Concatenate

from .nn_blocks import *

def discriminator_perceptually_aware(nc_in, input_size, vggface_res50=None):    
    inp1 = Input(shape=(input_size, input_size, nc_in))
    inp2 = Input(shape=(input_size, input_size, nc_in))   
    
    assert not (vggface_res50 == None), "Perceptual model not found."
    perceptual_feats = vggface_res50(inp1)
    pf1_1 = resize_tensor(perceptual_feats[1], [input_size//8, input_size//8])
    pf1_2 = resize_tensor(perceptual_feats[2], [input_size//16, input_size//16])
    pf1_3 = resize_tensor(perceptual_feats[3], [input_size//32, input_size//32])
    perceptual_feats = vggface_res50(inp2)
    pf2_1 = resize_tensor(perceptual_feats[1], [input_size//8, input_size//8])
    pf2_2 = resize_tensor(perceptual_feats[2], [input_size//16, input_size//16])
    pf2_3 = resize_tensor(perceptual_feats[3], [input_size//32, input_size//32])
    
    x = Concatenate()([inp1, inp2])
    nc_base = 32
    for i in range(3):
        x = conv_block_d(x, int(nc_base*(2**(i))), False)
    x = res_block(x, 128)
    x = Concatenate()([x, conv_block_d(pf1_1, f=128, k=1, strides=1), conv_block_d(pf2_1, f=128, k=1, strides=1)])
    x = conv_block_d(x, 256, False)
    x = res_block(x, 256)
    x = Concatenate()([x, conv_block_d(pf1_2, f=128, k=1, strides=1), conv_block_d(pf2_2, f=256, k=1, strides=1)])
    x = conv_block_d(x, 512, False)
    x = Concatenate()([x, conv_block_d(pf1_3, f=128, k=1, strides=1), conv_block_d(pf2_3, f=256, k=1, strides=1)])
    out = Conv2D(1, kernel_size=4, use_bias=False, padding="same")(x)   
    return Model(inputs=[inp1, inp2], outputs=out) 

def discriminator(nc_in, input_size=224):    
    inp = Input(shape=(input_size, input_size, nc_in))
    inp_segm = Input(shape=(input_size, input_size, nc_in))
    x = Concatenate()([inp, inp_segm])
    
    x = conv_block_d(x, 32, False)
    x = conv_block_d(x, 32, False)
    x = conv_block_d(x, 64, True)
    x = conv_block_d(x, 128, True)
    x = res_block(x, 128)
    x = conv_block_d(x, 256, True)
    x = res_block(x, 256)
    x = conv_block_d(x, 512, False, strides=1)
    out = Conv2D(1, kernel_size=4, use_bias=False, padding="same")(x)   
    return Model(inputs=[inp, inp_segm], outputs=out) 
