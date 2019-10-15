import cv2
import numpy as np

from.image_dataset import minibatch

def create_dataset(config):
    return minibatch(config)
