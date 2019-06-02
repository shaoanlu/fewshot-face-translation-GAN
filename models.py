import cv2
import os
import numpy as np
from keras.layers import Input
from keras import backend as K

import networks.generator as gen

INPUT_SIZE = 224
LATENT_DIM = 512
NC_IN = 3

class FaceTranslationGANInferenceModel:
    def __init__(self):
        self.input_size = INPUT_SIZE
        self.latent_dim = LATENT_DIM
        self.nc_in = NC_IN
        
        self.encoder = self.build_encoder()
        self.decoder = self.build_decoder()        
        try:
            self.encoder.load_weights(f"weights/encoder.h5")
            self.decoder.load_weights(f"weights/decoder.h5")
            print("Found checkpoints in weights folder. Built model with pre-trained weights.")
        except:
            print("Model built with default initializaiton.")
            pass
        
        image_size = (self.input_size, self.input_size, self.nc_in)
        inp_src = Input(shape=image_size)
        inp_tar = Input(shape=image_size)
        inp_segm = Input(shape=image_size)
        inp_emb = Input((self.latent_dim,))
        self.path_inference = K.function(
            [inp_src, inp_tar, inp_segm, inp_emb], 
            [self.decoder(self.encoder([inp_src, inp_tar, inp_segm]) + [inp_emb])]
        )
    def load_weights(self, weights_path):
        self.encoder.load_weights(os.path.join(weights_path, "encoder.h5"))
        self.decoder.load_weights(os.path.join(weights_path, "decoder.h5"))
        
    def build_encoder(self):
        return gen.encoder(self.nc_in, self.input_size)
        
    def build_decoder(self):
        return gen.decoder(512, self.input_size//16, self.nc_in, self.latent_dim)
    
    def preprocess_input(self, im):
        im = cv2.resize(im, (self.input_size, self.input_size))
        return im / 255 * 2 - 1
    
    def inference(self, src, mask, tar, emb_tar):
        return self.path_inference(            
            [
                self.preprocess_input(src)[None, ...], 
                self.preprocess_input(tar)[None, ...], 
                self.preprocess_input(mask.astype(np.uint8))[None, ...],
                emb_tar
            ])
        