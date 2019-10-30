from keras.models import Model
from keras.layers import *

from .nn_blocks import *

def encoder(nc_in=3, input_size=224):
    inp = Input(shape=(input_size, input_size, nc_in))
    inp_ref = Input(shape=(input_size, input_size, nc_in))
    inp_segm_mask = Input(shape=(input_size, input_size, nc_in))
    x = concatenate([inp, inp_ref, inp_segm_mask])
    conv1 = conv_block(x, 64, use_norm=True, strides=2)
    conv2 = conv_block(conv1, 128, use_norm=True, strides=2)
    conv3 = conv_block(conv2, 256, use_norm=True, strides=2)
    conv4 = conv_block(conv3, 512, use_norm=True, strides=2)
    return Model(
        [inp, inp_ref, inp_segm_mask], 
        [
            conv4, 
            resize_tensor(inp, [input_size//2, input_size//2]), 
            resize_tensor(inp, [input_size//4, input_size//4]),
            inp_segm_mask
        ]
    )

def decoder(nc_conv_in=512, input_size=14, nc_in=3, num_fc=3, latent_dim=512, sep_mean_var=False, additional_emb=False, use_nwg=False):
    """
        Arguments:
            nc_conv_in: int. Number of input channels. Should be consistent with number of encoder output channels
            input_size: int. Base resolution of input feature map
            nc_in: int. Number of channels of input RGB image
            num_fc: int. Number of fully connected layers apply on laten embeddings
            sep_mean_var. Boolean, whether to use separate mean/var params in AdaIN resblock
            additional_emb: Boolean, whether the decoder take additional embeddings of source face
            use_nwg: Boolean, whether to use network weight generation (proposed in few-shot vid2vid) module to replace SPADE.
    """
    conv4 = Input(shape=(input_size, input_size, nc_conv_in))
    inp_segm_mask = Input(shape=(input_size, input_size, nc_in))
    inp_ds2 = Input(shape=(input_size*8, input_size*8, nc_in))
    inp_ds4 = Input(shape=(input_size*4, input_size*4, nc_in))
    if additional_emb:
        inp_emb1 = Input(shape=(latent_dim,))
        inp_emb2 = Input(shape=(latent_dim,))
        inp_emb = Concatenate()([inp_emb1, inp_emb2])
        nc_emb_reshape = latent_dim*2
    else:
        inp_emb = Input(shape=(latent_dim,))
        nc_emb_reshape = latent_dim
    
    if use_nwg:
        emb_fc1 = embddding_fc_block(inp_emb, 2)
        emb_fc2 = embddding_fc_block(emb_fc1, 1)
        emb_fc3 = embddding_fc_block(emb_fc2, 1)
        emb_mean_var = embddding_fc_block(inp_emb, num_fc)
        emb_mean_var = Reshape((1, 1, 256))(emb_mean_var)
        emb = Reshape((1, 1, nc_emb_reshape))(inp_emb)
        emb = UpSampling2D((input_size,input_size))(emb)
        
        x = adain_resblock(conv4, emb_mean_var, 512, sep_mean_var) 
        x = adain_resblock(x, emb_mean_var, 512, sep_mean_var)   
        x = concatenate([x, emb])
        x = upscale_nn(x, 512)
        y, p_s0 = interm_img_syn_network(x, emb_fc1, resize_tensor(inp_segm_mask, [input_size*2, input_size*2]), f=512)
        x = concatenate([x, y])
        x = upscale_nn(x, 256)
        y, p_s1 = interm_img_syn_network(x, emb_fc2, UpSampling2D()(p_s0), f=256)
        x = concatenate([x, y, inp_ds4])
        x = upscale_nn(x, 128)
        x = concatenate([x, inp_ds2])
        x = upscale_nn(x, 64)
    else:
        emb_mean_var = embddding_fc_block(inp_emb, num_fc)
        emb_mean_var = Reshape((1, 1, 256))(emb_mean_var)
        emb = Reshape((1, 1, nc_emb_reshape))(inp_emb)
        emb = UpSampling2D((input_size,input_size))(emb)
        
        x = adain_resblock(conv4, emb_mean_var, 512, sep_mean_var) 
        x = adain_resblock(x, emb_mean_var, 512, sep_mean_var)   
        x = concatenate([x, emb])
        x = upscale_nn(x, 512)
        x = SPADE_res_block(x, resize_tensor(inp_segm_mask, [input_size*2, input_size*2]), 512, block_id='3')
        x = upscale_nn(x, 256)
        x = SPADE_res_block(x, resize_tensor(inp_segm_mask, [input_size*4, input_size*4]), 256, block_id='4')
        x = concatenate([x, inp_ds4])
        x = upscale_nn(x, 128)
        x = concatenate([x, inp_ds2])
        x = upscale_nn(x, 64)
    
    out = Conv2D(3, kernel_size=4, padding='same', activation="tanh")(x)
    if additional_emb:
        return Model([conv4, inp_ds2, inp_ds4, inp_segm_mask, inp_emb1, inp_emb2], out)
    else:
        return Model([conv4, inp_ds2, inp_ds4, inp_segm_mask, inp_emb], out)
