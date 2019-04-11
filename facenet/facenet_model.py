from facenet.facenet.inception_resnet_v1 import InceptionResNetV1
from keras.layers import Lambda, Input
from keras.models import Model
from keras import backend as K
import tensorflow as tf
import numpy as np


class FaceNet():
    def __init__(self, resolution=None, weights_path='./facenet/facenet/facenet_keras_weights_VGGFace2.h5', classes=512):
        self.weights_path = weights_path
        self.input_resolution = 160
        self.latent_dim = classes
        
        self.net = self.build_networks(resolution, classes=classes)        
        self.net.trainable = False
        
    def build_networks(self, resolution=None, classes=128):
        """
        FaceNet().net expects input images that have pixels normalized to [-1, +1].
        """
        input_tensor = Input((resolution, resolution, 3))
        facenet = InceptionResNetV1(weights_path=self.weights_path, classes=classes)
        facenet = Model(facenet.inputs, facenet.layers[-1].output) # layers[-1] is a BN layers
        rescale_layer = self.rescale()
        preprocess_layer = self.preprocess()
        l2_normalize = self.l2_norm()
        output_tensor = l2_normalize(facenet(preprocess_layer(rescale_layer(input_tensor))))        
        return Model(input_tensor, output_tensor)
        
    def preprocess(self):        
        def preprocess_facenet(x):
            """ 
                tf.image.per_image_standardization
                K.mean & K.std axis being [-3,-2,-1] or [-1,-2,-3] does nor affect output
                since the output shape is [batch_size, 1, 1, 1].
            """
            x = (x - 127.5) / 128
            x = K.map_fn(lambda im: tf.image.per_image_standardization(im), x)
            return x     
        
        input_tensor = Input((None, None, 3))      
        x = Lambda(
            lambda x: tf.image.resize_bilinear(
                x, 
                [self.input_resolution, self.input_resolution]
            ))(input_tensor)
        output_tensor = Lambda(preprocess_facenet)(x)        
        return Model(input_tensor, output_tensor)
    
    def rescale(self):
        """
            Scale image fomr range [-1, +1] to [0, 255].
        """
        def scale(x):
            return (x + 1) / 2 * 255
        
        input_tensor = Input((None, None, 3))  
        output_tensor = Lambda(scale)(input_tensor)
        return Model(input_tensor, output_tensor)
    
    def l2_norm(self):            
        input_tensor = Input((self.latent_dim,))
        output_tensor = Lambda(lambda x: K.l2_normalize(x))(input_tensor)
        return Model(input_tensor, output_tensor)
        
        