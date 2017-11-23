from __future__ import division
from __future__ import print_function

import prettytensor as pt
import tensorflow as tf
import misc.custom_ops
from misc.config import cfg

class VAEMnist(object):

    def __init__(self, image_shape, activation_function=tf.nn.elu, num_neurons_mlp=128,
                 gf_dim=128, z_dim=100, y_dim=None):
        self.activation_function = activation_function
        self.activation_function = activation_function

        self.input_height = cfg.TRAIN.INPUT_HEIGHT
        self.input_width = cfg.TRAIN.INPUT_WIDTH
        self.output_height = cfg.TRAIN.OUTPUT_HEIGHT
        self.output_width = cfg.TRAIN.OUTPUT_WIDTH

        self.y_dim = y_dim
        self.z_dim = z_dim

        self.gf_dim = gf_dim
        self.g_dim = gf_dim
        self.input_dim =784
        self.batch_size = cfg.TRAIN.BATCH_SIZE

        self.s = image_shape[0]
        self.image_shape = image_shape
        self.s2, self.s4, self.s8, self.s16 = \
            int(self.s / 2), int(self.s / 4), int(self.s / 8), int(self.s / 16)

        self.num_neurons_mlp = num_neurons_mlp
        self.z_size = z_dim

    def encoder(self, image):
        pre_z_image = \
            (pt.wrap(image).
             custom_fully_connected(512).
             apply(self.activation_function).
             custom_fully_connected(256).
             apply(self.activation_function).
             custom_fully_connected(128).
             apply(self.activation_function))
        image_mean = \
            (pt.wrap(pre_z_image).
             custom_fully_connected(self.z_size))
        image_log_sigma_sq = \
            (pt.wrap(pre_z_image).
             custom_fully_connected(self.z_size))
        return image_mean, image_log_sigma_sq

    def decoder(self, z_var):
        image = \
            (pt.wrap(z_var).
             custom_fully_connected(128).
             apply(self.activation_function).
             custom_fully_connected(256).
             apply(self.activation_function).
             custom_fully_connected(512).
             apply(self.activation_function).
             custom_fully_connected(self.input_dim).
             apply(tf.sigmoid))
        return image

    def get_mean_stddev(self, image=None):
        mean, log_sigma_sq = self.encoder(image)
        ret_list = [mean, log_sigma_sq]
        return ret_list

    def get_reconstructed_image(self, latent_space):
        reconstructed_image = self.decoder(latent_space)
        return reconstructed_image
