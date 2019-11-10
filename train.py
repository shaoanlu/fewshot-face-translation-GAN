from keras.optimizers import Adam, Adamax, RMSprop
import keras.backend as K
import numpy as np

from models import FaceTranslationGANTrainModel
from data import create_dataset
from utils.config_loader import load_yaml

class Trainer:
    def __init__(self, model, config):        
        self.model = model
        self.batch_size = int(config['batch_size'])
        self.input_size = int(config["input_size"])
        self.lr_gen = config['lr_gen']
        self.lr_dis = config['lr_dis']

        self.build_training_functions()
        self.clear_loss()

    def clear_loss(self):
        self.loss_gen = [0 for _ in range(self.num_update_outputs_gen)]
        self.loss_dis = [0 for _ in range(self.num_update_outputs_dis)]

    def train_one_iter(self, data1, data2):
        assert isinstance(data1, tuple), type(data1)
        assert isinstance(data2, tuple), type(data2)

        loss = self.dis_train(data1)
        for i, l in enumerate(loss):
            self.loss_dis[i] += l

        loss = self.gen_train(data2)
        for i, l in enumerate(loss):
            self.loss_gen[i] += l

    def build_training_functions(self):
        weights_gen = self.model.get_generator_trainable_weights()
        ttl_loss_gen = self.model.get_generator_total_loss()
        inp_tensors_gen, out_tensors_gen = self.model.get_generator_update_tensors()
        self.num_update_outputs_gen = len(out_tensors_gen)
        training_updates_gen = Adam(lr=self.lr_gen, beta_1=0.5, clipvalue=0.5).get_updates(weights_gen, [], ttl_loss_gen)
        self.gen_train = K.function(
            inp_tensors_gen, 
            out_tensors_gen, 
            training_updates_gen)
        
        weights_dis = self.model.get_discriminator_trainable_weights()
        ttl_loss_dis = self.model.get_discriminator_total_loss()
        inp_tensors_dis, out_tensors_dis = self.model.get_discriminator_update_tensors()
        self.num_update_outputs_dis = len(out_tensors_dis)
        training_updates_dis = Adam(lr=self.lr_dis, beta_1=0.5, clipvalue=0.5).get_updates(weights_dis, [], ttl_loss_dis)
        self.dis_train = K.function(
            inp_tensors_dis, 
            out_tensors_dis, 
            training_updates_dis)

    def show_current_loss(self, iter):
        print(f"[Iteration {str(iter)}]")
        print("Loss G:")
        print(" | ".join(self.loss_gen))
        print("Loss D:")
        print(" | ".join(self.loss_dis))

    def save_weights(self, dir_weights="weights"):
        self.model.save_weights(dir_weights=dir_weights)

    def display_current_result(self, data):
        raise NotImplementedError()
    
    def save_preview_image(self, data_random, data_fixed, iter, save_path="preview"):
        """
        Two images will be saved in this function:
            1. An image showing results of random sampled source/target pairs.
            2. An image showing results of fixed (throughout the entire training) source/target pairs.
        """
        def to_uint8(x):
            x = (x + 1) / 2 * 255
            return x.astype(np.uint8)

        result1 = self.model.path_inference([data_random[0], data_random[4], data_random[1]])
        result2 = self.model.path_inference([data_fixed[0], data_fixed[4], data_fixed[1]])

        out1 = np.zeros((self.batch_size*self.input_size, 4*self.input_size, 3))
        out2 = np.zeros((self.batch_size*self.input_size, 4*self.input_size, 3))
        for i in range(self.batch_size):
            out1[i*self.input_size:(i+1)*self.input_size, 0:self.input_size, :] = to_uint8(data_random[0][i, ...])
            out1[i*self.input_size:(i+1)*self.input_size, 1*self.input_size:2*self.input_size, :] = to_uint8(data_random[1][i, ...])
            out1[i*self.input_size:(i+1)*self.input_size, 2*self.input_size:3*self.input_size, :] = to_uint8(data_random[4][i, ...])
            out1[i*self.input_size:(i+1)*self.input_size, 3*self.input_size:4*self.input_size, :] = to_uint8(result1[0][i, ...])
        for i in range(self.batch_size):
            out2[i*self.input_size:(i+1)*self.input_size, 0:self.input_size, :] = to_uint8(data_fixed[0][i, ...])
            out2[i*self.input_size:(i+1)*self.input_size, 1*self.input_size:2*self.input_size, :] = to_uint8(data_fixed[1][i, ...])
            out2[i*self.input_size:(i+1)*self.input_size, 2*self.input_size:3*self.input_size, :] = to_uint8(data_fixed[4][i, ...])
            out2[i*self.input_size:(i+1)*self.input_size, 3*self.input_size:4*self.input_size, :] = to_uint8(result2[0][i, ...])
        cv2.imwrite(str(PurePath(save_path, "random_result_iter{str(iter)}.jpg")), out1[..., ::-1])
        cv2.imwrite(str(PurePath(save_path, "fixed_result_iter{str(iter)}.jpg")), out2[..., ::-1])

if __name__ == "__main__":
    path_config = "configs/config.yaml"
    config = load_yaml(path_config)
    dataset = create_dataset(config) 

    model = FaceTranslationGANTrainModel(config)
    trainer = Trainer(model=model, config=config)

    max_iter = config["max_iter"]
    save_iter = config["save_iter"]
    anchor_batch = next(dataset) # images for fixed preview
    trainer.clear_loss()
    for iteration in range(max_iter):
        data_trn_d = next(dataset)
        data_trn_g = next(dataset)
        trainer.train_one_iter(data_trn_d, data_trn_g)
        if (iteration + 1) % save_iter == 0:
            data_infer = next(dataset)
            trainer.show_current_loss(iteration)
            trainer.save_preview_image(data_infer, anchor_batch, iteration)
            trainer.save_weights()
            trainer.clear_loss()


