#-*- coding: utf-8 -*-
from __future__ import division
import os
import time
import tensorflow as tf
import numpy as np
import random 

# from ops import *
from utils import *

slim = tf.contrib.slim
layers = tf.contrib.layers

def lrelu(x, leak=0.2, name="lrelu"):
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * abs(x)

class WGAN(object):
    model_name = "WGAN"     # name for checkpoint

    def __init__(self, sess, epoch, batch_size, z_dim, dataset_name, checkpoint_dir, result_dir, log_dir, mode):
        self.sess = sess
        self.dataset_name = dataset_name
        self.checkpoint_dir = checkpoint_dir
        self.result_dir = result_dir
        self.log_dir = log_dir
        self.epoch = epoch
        self.batch_size = batch_size

        if dataset_name == 'mnist' or dataset_name == 'fashion-mnist':
            # parameters
            self.input_height = 28
            self.input_width = 28
            self.output_height = 28
            self.output_width = 28

            self.z_dim = z_dim         # dimension of noise-vector
            self.c_dim = 1

            # WGAN parameter
            self.disc_iters = 1     # The number of critic iterations for one-step of generator

            # train
            self.learning_rate = 0.0002
            self.beta1 = 0.5

            # test
            self.sample_num = 64  # number of generated images to be saved

            # load mnist
            self.data_X, self.data_Y = load_mnist(self.dataset_name)

            # get number of batches for a single epoch
            self.num_batches = len(self.data_X) // self.batch_size
        elif dataset_name == 'Anime':
            # parameters
            self.input_height = 64
            self.input_width = 64
            self.output_height = 64
            self.output_width = 64

            self.z_dim = z_dim          # dimension of noise-vector
            self.y_dim = 23             # dimension of condition-vector (label)
            self.c_dim = 3
        
            print ('[Tip 14] Train discriminator more (sometimes)')
            # Tip 14. [notsure] Train discriminator more (sometimes)
            #   Especially when you have noise.
            #   Hard to find a schedule of number of D iterations vs G iterations.

            # WGAN parameter
            self.d_iters = 2            # The number of critic iterations for each epoch
            self.g_iters = 1            # The number of critic iterations for each epoch
            
            # train
            self.learning_rate = 0.0001
            self.beta1 = 0.5
            self.beta2 = 0.9

            # test
            self.sample_num = 25  # number of generated images to be saved

            if mode == 'train':
                # load Anime
                self.data_X, self.data_Y = load_Anime('./extra_data/images/')

                # get number of batches for a single epoch
                self.num_batches = len(self.data_X) // self.batch_size
        else:
            raise NotImplementedError

    # Tip 5. Avoid Sparse Gradients: ReLU, MaxPool
    #   The stability of the GAN game suffers if you have sparse gradients.
    #   LeakyReLU = good (in both G and D).
    #   For Downsampling, use: Average Pooling, Conv2d + stride.
    #   For Upsampling, use: PixelShuffle, ConvTranspose2d + stride.
    #       PixelShuffle: https://arxiv.org/abs/1609.05158
    def discriminator(self, x, y, is_training=True, reuse=False):
        print ('[Tip 5] Avoid Sparse Gradients: ReLU, MaxPool')
        with tf.variable_scope("discriminator", reuse=reuse):
            with slim.arg_scope(
                [layers.conv2d, layers.fully_connected],
                activation_fn=None, normalizer_fn=None,
                weights_initializer=tf.truncated_normal_initializer(stddev=0.02)
                ):

                    net = lrelu(layers.conv2d(x, 64, [5, 5], stride=2))
                    net = lrelu(layers.conv2d(net, 128, [5, 5], stride=2))
                    net = lrelu(layers.conv2d(net, 256, [5, 5], stride=2))
                    net = lrelu(layers.conv2d(net, 384, [5, 5], stride=2))

                    # embed_y = tf.expand_dims(y, 1)
                    # embed_y = tf.expand_dims(embed_y, 2)
                    # tiled_embeddings = tf.tile(embed_y, [1, self.input_width // 16, self.output_height // 16, 1])

                    # h3_concat = tf.concat([net, tiled_embeddings], axis=-1)
                    h3_concat = tf.concat([net], axis=-1)

                    net = lrelu(layers.conv2d(h3_concat, 384, [1, 1], stride=1, padding='valid'))
                    net = layers.flatten(net)
                    # net = layers.fully_connected(net, (self.input_width // 8) * (self.output_height // 8) * 256)
                    net = layers.fully_connected(net, 1, normalizer_fn=None, activation_fn=None)
        return net

    def generator(self, z, y, is_training=True, reuse=False):
        print ('[Tip 5] Avoid Sparse Gradients: ReLU, MaxPool')
        with tf.variable_scope("generator", reuse=reuse):
            with slim.arg_scope(
                [layers.fully_connected, layers.conv2d_transpose],
                activation_fn=None, normalizer_fn=None,
                weights_initializer=tf.truncated_normal_initializer(stddev=0.02),
                ):
                    # x = tf.concat([z, y], axis=-1)
                    x = tf.concat([z], axis=-1)
                    net = tf.layers.batch_normalization(
                        layers.fully_connected(x,  (self.input_width // 16) * (self.output_height // 16) * 384),
                        training=is_training, momentum=0.9, epsilon=1e-5)
                    net = tf.reshape(net, [-1, self.input_width // 16, self.output_height // 16, 384])
                    net = tf.nn.relu(tf.layers.batch_normalization(
                            layers.conv2d_transpose(net, 256, [5, 5], stride=2), 
                            training=is_training, momentum=0.9, epsilon=1e-5))
                    net = tf.nn.relu(tf.layers.batch_normalization(
                            layers.conv2d_transpose(net, 128, [5, 5], stride=2), 
                            training=is_training, momentum=0.9, epsilon=1e-5))
                    net = tf.nn.relu(tf.layers.batch_normalization(
                            layers.conv2d_transpose(net, 64, [5, 5], stride=2), 
                            training=is_training, momentum=0.9, epsilon=1e-5))
                    net = layers.conv2d_transpose(net, self.c_dim, [5, 5], stride=2, normalizer_fn=None, activation_fn=None) 
                    net = tf.nn.tanh(net)
        return net

    def build_model(self):
        # Fix random seed.
        np.random.seed(9487)
        random.seed(9487)
        tf.set_random_seed(9487)

        # some parameters
        image_dims = [self.input_height, self.input_width, self.c_dim]
        bs = self.batch_size

        """ Graph Input """
        # images
        self.inputs = tf.placeholder(tf.float32, [bs] + image_dims, name='real_images')
        print ('[Tip Hung-yi Lee] Data Augmentation (flip images horizontally + rotate images)')
        # Randomly flip images horizontally (left to right).
        images_filped = tf.map_fn(lambda img: tf.image.random_flip_left_right(img), self.inputs)
        # Random rotation angles between -15 and 15 degree.
        angles = tf.random_uniform([bs], minval=-15.0 * np.pi / 180.0, maxval=15.0 * np.pi / 180.0, dtype=tf.float32, seed=9487)
        # Randomly rotate images between -15 and 15 degree.
        images_rotated = tf.contrib.image.rotate(images_filped, angles, interpolation='NEAREST')

        # labels
        self.y = tf.placeholder(tf.float32, [bs, self.y_dim], name='y')

        # noises
        self.z = tf.placeholder(tf.float32, [bs, self.z_dim], name='z')

        # # wrong images
        # self.imgs_wrong = tf.placeholder(tf.float32, [bs] + image_dims, name='wrong_images')

        # # wrong labels
        # self.labels_wrong = tf.placeholder(tf.float32, [bs, self.y_dim], name='wrong_y')
        """ Loss Function """

        print ('[Tip 4] BatchNorm')
        # Tip 4. BatchNorm
        #   Construct different mini-batches for real and fake, i.e. each mini-batch needs to contain only all real images or all generated images.
        #   When batchnorm is not an option use instance normalization (for each sample, subtract mean and divide by standard deviation).

         # Output of D for fake images.
        G = self.generator(self.z, self.y, is_training=True, reuse=False)
        D_fake = self.discriminator(G, self.y, is_training=True, reuse=tf.AUTO_REUSE)
        # Output of D for real images.
        D_real = self.discriminator(images_rotated, self.y, is_training=True, reuse=tf.AUTO_REUSE)
        # # Output of D for wrong images.
        # D_wrong_img = self.discriminator(self.imgs_wrong, self.y, is_training=True, reuse=tf.AUTO_REUSE)
        # # Output of D for wrong labels.
        # D_wrong_label = self.discriminator(images_rotated, self.labels_wrong, is_training=True, reuse=tf.AUTO_REUSE)
       

        # get loss for discriminator
        d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_real, labels=tf.ones_like(D_real)))
        d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake, labels=tf.zeros_like(D_fake)))
                    #    tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_wrong_img, labels=tf.zeros_like(D_wrong_img))) + \
                    #    tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_wrong_label, labels=tf.zeros_like(D_wrong_label))) ) / 3

        self.d_loss = d_loss_real + d_loss_fake

        # get loss for generator
        self.g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake, labels=tf.ones_like(D_fake)))
        # self.g_loss = - d_loss_fake

        """ Training """
        # divide trainable variables into a group for D and a group for G
        d_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'discriminator')
        g_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'generator')
        # t_vars = tf.trainable_variables()
        # d_vars = [var for var in t_vars if 'd_' in var.name]
        # g_vars = [var for var in t_vars if 'g_' in var.name]
        
        # weight clipping
        self.clip_D = [p.assign(tf.clip_by_value(p, -0.01, 0.01)) for p in d_vars]

        # optimizers
        self.d_optim = tf.train.AdamOptimizer(self.learning_rate, beta1=self.beta1, beta2=self.beta2) \
                      .minimize(self.d_loss, var_list=d_vars)

        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            self.g_optim = tf.train.AdamOptimizer(self.learning_rate, beta1=self.beta1, beta2=self.beta2) \
                      .minimize(self.g_loss, var_list=g_vars)

        """" Testing """
        # for test
        self.fake_images = self.generator(self.z, self.y, is_training=False, reuse=True)

        """ Summary """
        d_loss_real_sum = tf.summary.scalar("d_loss_real", d_loss_real)
        d_loss_fake_sum = tf.summary.scalar("d_loss_fake", d_loss_fake)
        d_loss_sum = tf.summary.scalar("d_loss", self.d_loss)
        g_loss_sum = tf.summary.scalar("g_loss", self.g_loss)

        # final summary operations
        self.g_sum = tf.summary.merge([d_loss_fake_sum, g_loss_sum])
        self.d_sum = tf.summary.merge([d_loss_real_sum, d_loss_sum])

    def train(self):
        # Fix random seed.
        np.random.seed(9487)
        random.seed(9487)
        tf.set_random_seed(9487)

        # initialize all variables
        tf.global_variables_initializer().run()

        # graph inputs for visualize training results
        # self.sample_z = np.random.uniform(-1, 1, size=(self.batch_size , self.z_dim))

        print ('[Tip 3] Use a spherical Z')
        # Tip 3. Use a spherical Z
        #   Dont sample from a Uniform distribution.
        #   Sample from a gaussian distribution.
        #   When doing interpolations, do the interpolation via a great circle, rather than a straight line from point A to point B.
        #   Tom White's Sampling Generative Networks ref code https://github.com/dribnet/plat has more details.
        self.sample_z = np.random.normal(0, np.exp(-1 / np.pi), size=(self.batch_size, self.z_dim))
        test_z_sample = np.random.normal(0, np.exp(-1 / np.pi), size=(self.batch_size, self.z_dim))

        # # self.test_labels = self.data_Y[0:self.batch_size]
        # self.test_labels = np.zeros((self.batch_size, 23))
        # # blue hair, blue eye.
        # for i in range(5):
        #     self.test_labels[0 + i][8] = 1
        #     self.test_labels[0 + i][22] = 1
        # # blue hair, green eye.
        # for i in range(5):
        #     self.test_labels[5 + i][8] = 1
        #     self.test_labels[5 + i][19] = 1
        # # blue hair, red eye.
        # for i in range(5):
        #     self.test_labels[10 + i][8] = 1
        #     self.test_labels[10 + i][21] = 1
        # # green hair, blue eye.
        # for i in range(5):
        #     self.test_labels[15 + i][4] = 1
        #     self.test_labels[15 + i][22] = 1
        # # green hair, red eye.
        # for i in range(5):
        #     self.test_labels[20 + i][4] = 1
        #     self.test_labels[20 + i][21] = 1

        # saver to save model
        self.saver = tf.train.Saver(max_to_keep=50)

        # summary writer
        self.writer = tf.summary.FileWriter(self.log_dir + '/' + self.model_name, self.sess.graph)

        # restore check-point if it exits
        could_load, checkpoint_counter = self.load(self.checkpoint_dir)
        if could_load:
            start_epoch = (int)(checkpoint_counter / self.num_batches)
            start_batch_id = checkpoint_counter - start_epoch * self.num_batches
            counter = checkpoint_counter
            print(" [*] Load SUCCESS")
        else:
            start_epoch = 0
            start_batch_id = 0
            counter = 1
            print(" [!] Load failed...")

        # loop for epoch
        start_time = time.time()
        g_loss = 0.0
        for epoch in range(start_epoch, self.epoch):

            # get batch data
            for idx in range(start_batch_id, self.num_batches):
                batch_images = np.asarray(self.data_X[idx * self.batch_size:(idx + 1) * self.batch_size]).astype(np.float32)
                batch_labels = np.asarray(self.data_Y[idx * self.batch_size:(idx + 1) * self.batch_size]).astype(np.float32)
                # batch_images_wrong = np.asarray(self.data_X[random.sample(range(len(self.data_X)), len(batch_images))]).astype(np.float32)
                # batch_labels_wrong = np.asarray(self.data_Y[random.sample(range(len(self.data_Y)), len(batch_images))]).astype(np.float32)
                # print(batch_images.shape)
                # print(batch_images_wrong.shape)
                # print(batch_labels.shape)
                # print(batch_labels_wrong.shape)
                # batch_z = np.random.uniform(-1, 1, [self.batch_size, self.z_dim]).astype(np.float32)

                # Tip 3. Use a spherical Z
                #   Dont sample from a Uniform distribution.
                #   Sample from a gaussian distribution.
                #   When doing interpolations, do the interpolation via a great circle, rather than a straight line from point A to point B.
                #   Tom White's Sampling Generative Networks ref code https://github.com/dribnet/plat has more details.
                batch_z = np.random.normal(0, np.exp(-1 / np.pi), [self.batch_size, self.z_dim]).astype(np.float32)

                # Tip 14. [notsure] Train discriminator more (sometimes)
                #   Especially when you have noise.
                #   Hard to find a schedule of number of D iterations vs G iterations.

                # update D network
                for _ in range(self.d_iters):
                    _, _, summary_str, d_loss = self.sess.run([self.d_optim, self.clip_D, self.d_sum, self.d_loss],
                                                # feed_dict={self.inputs: batch_images, self.y: batch_labels, self.z: batch_z, self.imgs_wrong: batch_images_wrong, self.labels_wrong: batch_labels_wrong})
                                                feed_dict={self.inputs: batch_images, self.z: batch_z})
                    self.writer.add_summary(summary_str, counter)

                # batch_z = np.random.uniform(-1, 1, [self.batch_size, self.z_dim]).astype(np.float32)

                # Tip 3. Use a spherical Z
                #   Dont sample from a Uniform distribution.
                #   Sample from a gaussian distribution.
                #   When doing interpolations, do the interpolation via a great circle, rather than a straight line from point A to point B.
                #   Tom White's Sampling Generative Networks ref code https://github.com/dribnet/plat has more details.
                batch_z = np.random.normal(0, np.exp(-1 / np.pi), [self.batch_size, self.z_dim]).astype(np.float32)

                # update G network
                # if (counter-1) % self.disc_iters == 0:
                for _ in range(self.g_iters):
                    _, summary_str, g_loss = self.sess.run([self.g_optim, self.g_sum, self.g_loss],
                                                # feed_dict={self.inputs: batch_images, self.y: batch_labels, self.z: batch_z, self.imgs_wrong: batch_images_wrong, self.labels_wrong: batch_labels_wrong})
                                                feed_dict={self.inputs: batch_images, self.z: batch_z})
                    self.writer.add_summary(summary_str, counter)

                counter += 1

                # display training status
                print("Epoch: [%4d/%4d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f" \
                      % (epoch, self.epoch, idx, self.num_batches, time.time() - start_time, d_loss, g_loss), end='\r')

                # # save training results for every 300 steps
                # if np.mod(counter, 300) == 0:
                #     samples = self.sess.run(self.fake_images,
                #                             feed_dict={self.z: self.sample_z})
                #     tot_num_samples = min(self.sample_num, self.batch_size)
                #     manifold_h = int(np.floor(np.sqrt(tot_num_samples)))
                #     manifold_w = int(np.floor(np.sqrt(tot_num_samples)))
                #     save_images(samples[:manifold_h * manifold_w, :, :, :], [manifold_h, manifold_w],
                #                 './' + check_folder(self.result_dir + '/' + self.model_dir) + '/' + self.model_name + '_train_{:02d}_{:04d}.png'.format(
                #                     epoch, idx))

            print ()

            # After an epoch, start_batch_id is set to zero
            # non-zero value is only for the first epoch after loading pre-trained model
            start_batch_id = 0

            # save model
            self.save(self.checkpoint_dir, counter)

            # show temporal results
            self.visualize_results(epoch, test_z_sample)

        # save model for final step
        self.save(self.checkpoint_dir, counter)

    def visualize_results(self, epoch, z_sample):
        tot_num_samples = min(self.sample_num, self.batch_size)
        image_frame_dim = int(np.floor(np.sqrt(tot_num_samples)))

        """ random condition, random noise """

        # z_sample = np.random.uniform(-1, 1, size=(self.batch_size, self.z_dim))
        # z_sample = np.random.normal(0, 1, size=(self.batch_size, self.z_dim))

        samples = self.sess.run(self.fake_images, feed_dict={self.z: z_sample})

        save_images(samples[:image_frame_dim * image_frame_dim, :, :, :], [image_frame_dim, image_frame_dim],
                    check_folder(self.result_dir + '/' + self.model_dir) + '/' + self.model_name + '_epoch%03d' % epoch + '_test_all_classes.png')
    
    def infer(self):
        # Fix random seed.
        np.random.seed(9487)
        random.seed(9487)
        tf.set_random_seed(9487)

        # initialize all variables
        tf.global_variables_initializer().run()
        
        # saver to save model
        self.saver = tf.train.Saver(max_to_keep=50)

        # restore check-point if it exits
        could_load, checkpoint_counter = self.load(self.checkpoint_dir)
        if could_load:
            image_frame_dim = int(np.floor(np.sqrt(self.sample_num)))
            """ random condition, random noise """

            # Tip 3. Use a spherical Z
            #   Dont sample from a Uniform distribution.
            #   Sample from a gaussian distribution.
            #   When doing interpolations, do the interpolation via a great circle, rather than a straight line from point A to point B.
            #   Tom White's Sampling Generative Networks ref code https://github.com/dribnet/plat has more details.

            z_sample = np.random.normal(0, np.exp(-1 / np.pi), size=(self.batch_size, self.z_dim))
            z_sample = np.random.normal(0, np.exp(-1 / np.pi) / (10 / (2 * np.pi)), size=(self.batch_size, self.z_dim))
            z_sample = np.random.uniform(-np.exp(-1 / np.pi), np.exp(-1 / np.pi), size=(self.batch_size, self.z_dim))
            # z_sample = np.random.normal(1 / np.pi, np.exp(-1 / np.pi), size=(self.batch_size, self.z_dim))

            samples = self.sess.run(self.fake_images, feed_dict={self.z: z_sample})

            save_images(samples[:image_frame_dim * image_frame_dim, :, :, :], [image_frame_dim, image_frame_dim],
                        check_folder('./samples') + '/' + 'gan.png')
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")
            

    @property
    def model_dir(self):
        return "{}_{}_{}_{}".format(
            self.model_name, self.dataset_name,
            self.batch_size, self.z_dim)

    def save(self, checkpoint_dir, step):
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir, self.model_name)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,os.path.join(checkpoint_dir, self.model_name+'.model'), global_step=step)

    def load(self, checkpoint_dir):
        import re
        print(" [*] Reading checkpoints...")
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir, self.model_name)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            counter = int(next(re.finditer("(\d+)(?!.*\d)",ckpt_name)).group(0))
            print(" [*] Success to read {}".format(ckpt_name))
            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0