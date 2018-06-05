from __future__ import division
from __future__ import print_function

import numpy as np
import sys
from six.moves import range
from progressbar import ETA, Bar, Percentage, ProgressBar
from misc.config import cfg
import misc.utils as utils
import cv2
import prettytensor as pt
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

class VAEMnsitTrainer(object):

    def __init__(self, model, exp_name="mnist_vae"):
        self.model = model
        self.exp_name = exp_name
        self.save_dir = cfg.TRAIN.MNIST_VAE_DIR
        self.vae_learning_rate = cfg.TRAIN.VAE_LEARNING_RATE
        self.vae_model_path = cfg.TRAIN.VAE_PRETRAINED_MODEL
        self.image_shape = [28, 28, 1]
        self.batch_size = 64
        self.snapshot_interval = cfg.TRAIN.SNAPSHOT_INTERVAL
        self.log_dir = cfg.TRAIN.VAEMNIST_LOG_DIR
        self.dataset = input_data.read_data_sets("MNIST_data/", one_hot=True)
        self.input_dim = list(self.dataset.train.images[0].shape)

    def build_placeholder(self):
        self.input_images = tf.placeholder(tf.float32, [self.batch_size] + self.input_dim)
        self.latent_space = tf.placeholder(tf.float32)
        self.reconstructed_images = tf.placeholder(tf.float32, [self.batch_size] + self.input_dim)

    def init_op(self):
        self.build_placeholder()
        with pt.defaults_scope(phase=pt.Phase.train):
            with tf.variable_scope("vae_mnist_model"):
                vae_loss = self.compute_losses(self.input_images)
                print("VAE losses computed")
                self.prepare_trainer(loss=vae_loss)

    def kl_loss(self, mu, log_sigma):
        with tf.name_scope("KL_divergence"):
            loss = -log_sigma + 0.5 * (-1 + tf.exp(2. * log_sigma) + tf.square(mu))
            loss = tf.reduce_mean(loss)
            return loss

    def compute_losses(self, image):
        mean, log_sigma_sq = self.model.get_mean_stddev(image)
        eps = tf.random_normal(tf.shape(log_sigma_sq), 0, 1, dtype=tf.float32)
        self.latent_space = mean + tf.sqrt(tf.exp(log_sigma_sq)) * eps
        self.reconstructed_image = self.model.get_reconstructed_image(self.latent_space)
        self.reconstructed_image = tf.clip_by_value(self.reconstructed_image, 1e-8, 1 - 1e-8)
        epsilon = 1e-10
        rec_loss = - tf.reduce_sum(image * tf.log(self.reconstructed_image + epsilon) + \
                                            (1 - image) * tf.log(epsilon + 1 - self.reconstructed_image), axis=1)
        self.reconstruction_loss = tf.reduce_mean(rec_loss)
        latent_loss = -0.5 * tf.reduce_sum(
            1 + log_sigma_sq - tf.square(mean) - tf.exp(log_sigma_sq), axis=1
        )
        self.kl_div = tf.reduce_mean(latent_loss)
        elbo = tf.reduce_mean(self.reconstruction_loss + self.kl_div)
        self.vae_loss = elbo
        return self.vae_loss

    def prepare_trainer(self, loss):
        vae_opt = tf.train.AdamOptimizer(self.vae_learning_rate)
        self.vae_trainer = \
            pt.apply_optimizer(vae_opt, losses=[loss])

    def build_model(self, sess):
        self.init_op()
        sess.run(tf.initialize_all_variables())

        if len(self.vae_model_path) > 0:
            print("Reading VAE model parameters from %s" % self.vae_model_path)
            restore_vars = tf.all_variables()
            saver = tf.train.Saver(restore_vars)
            saver.restore(sess, self.vae_model_path)
            istart = self.vae_model_path.rfind('_') + 1
            iend = self.vae_model_path.rfind('.')
            counter = self.vae_model_path[istart:iend]
            counter = int(counter)
        else:
            print("Created model with fresh parameters.")
            counter = 0
        return counter

    def train(self):
        print("Running training for VAE on MNIST dataset")
        config = tf.ConfigProto(allow_soft_placement=True)

        with tf.Session(config=config) as sess:
            self.session = sess
            with tf.device("/gpu:%d" % cfg.GPU_ID):
                counter = self.build_model(sess)
                saver = tf.train.Saver(tf.all_variables(), keep_checkpoint_every_n_hours=2)
                num_examples = self.dataset.train.num_examples
                updates_per_epoch = num_examples // self.batch_size
                epoch_start = counter // updates_per_epoch
                for epoch in range(epoch_start, 150+1):
                    widgets = ["epoch #%d|" % epoch,
                               Percentage(), Bar(), ETA()]
                    pbar = ProgressBar(maxval=updates_per_epoch,
                                       widgets=widgets)
                    pbar.start()
                    vae_loss = 0
                    for iter in range(updates_per_epoch):
                        input_image, input_y = self.dataset.train.next_batch(self.batch_size)
                        input_image = np.array(input_image)
                        # input_image = input_image.reshape((self.batch_size, 28, 28, 1))
                        feed_dict = {self.input_images: input_image}
                        feed_out = [self.vae_trainer, self.reconstructed_image,
                                    self.reconstruction_loss, self.kl_div,
                                    self.vae_loss, self.latent_space]
                        _, rec_img, rec_loss, kl_loss, curr_vae_loss, curr_latent_space = sess.run(feed_out, feed_dict)
                        vae_loss += curr_vae_loss

                        if iter % 500 == 0:
                            # print("Printing type of current latent space: " + str(type(curr_latent_space)))
                            eps = np.random.normal(loc=0, scale=1, size=(64, 100))
                            # curr_latent_space = curr_latent_space + eps
                            curr_feed_out = [self.reconstructed_image]
                            gen_img = sess.run(curr_feed_out, feed_dict={self.latent_space: eps})[0]

                            gen_img = utils.reshape_and_tile_images(gen_img * 255)
                            rec_img = utils.reshape_and_tile_images(rec_img * 255)
                            orig_img = utils.reshape_and_tile_images(input_image * 255)
                            gen_img_filename = self.save_dir + "/epoch_%d/%d_gen_img.jpg" % (epoch, iter)
                            rec_img_filename = self.save_dir + "/epoch_%d/%d_rec_img.jpg" % (epoch, iter)
                            orig_img_filename = self.save_dir + "/epoch_%d/%d_orig_img.jpg" % (epoch, iter)
                            utils.mkdir_p(self.save_dir + "/epoch_%d" % (epoch))
                            cv2.imwrite(rec_img_filename, rec_img)
                            cv2.imwrite(orig_img_filename, orig_img)
                            cv2.imwrite(gen_img_filename, gen_img)
                        counter += 1
                        if counter % self.snapshot_interval == 0:
                            snapshot_path = '%s/%s_%s.ckpt' %\
                                            (self.log_dir,
                                             self.exp_name,
                                             str(counter))
                            utils.mkdir_p(snapshot_path)
                            fn = saver.save(sess, snapshot_path)
                            print("Model saved in file: %s" % fn)
                    vae_loss = vae_loss // updates_per_epoch
                    log_line = "%s: %s, %s: %s, %s: %s" % ("vae loss", vae_loss, "reconstruction loss", rec_loss,
                                                           "kl loss", kl_loss)
                    print("Epoch %d | " % (epoch) + log_line)
                    sys.stdout.flush()
