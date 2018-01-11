from __future__ import print_function
import os

import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from src.utils import setup_updates, save, load, save_images, get_training_data
from src.ops import KL_Gaussian, match, make_z
from src.net import net_G, net_D1

class dc_GAN(object):
    def __init__ (self, sess, flags ):
        self.sess = sess
        self.flags = flags
        self.flags.noise = 'normal'

        self.generator = net_G(self.flags)
        self.discriminator_1 = net_D1(self.flags)
        self.batch_images = get_training_data(self.flags)
        self.build_model()

    def build_model(self):
        self.im = self.batch_images
        self.z = make_z(self.flags)

        self.im_sum = tf.summary.image('real_img', self.im)

        self.im_hat = self.generator.net(self.z)
        self.d1_real = self.discriminator_1.net(self.im)
        self.d1_fake = self.discriminator_1.net(self.im_hat, reuse = True)

        self.im_hat_sum = tf.summary.image('fake_img', self.im_hat)

        # generator loss
        # print (tf.ones_like(self.d1_fake).shape)
        # print (self.d1_fake.shape)
        # print ('**********')
        # raise
        self.g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            labels = tf.ones_like(self.d1_fake), logits = self.d1_fake
        ))
        self.g_loss_sum = tf.summary.scalar('g_loss', self.g_loss)

        # discriminator_1 loss
        self.d1_fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            labels = tf.zeros_like(self.d1_fake), logits = self.d1_fake
        ))
        self.d1_real_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            labels = tf.ones_like(self.d1_real), logits = self.d1_real
        ))
        self.d1_loss = self.d1_real_loss + self.d1_fake_loss
        self.d1_fake_loss_sum = tf.summary.scalar('d1_fake_loss', self.d1_fake_loss)
        self.d1_real_loss_sum = tf.summary.scalar('d1_real_loss', self.d1_real_loss)
        self.d1_loss_sum = tf.summary.scalar('d1_loss', self.d1_loss)


        trainable_vars = tf.trainable_variables()
        self.g_vars = [var for var in trainable_vars if 'g_' in var.name]
        self.d1_vars = [var for var in trainable_vars if 'd1_' in var.name]

        self.saver = tf.train.Saver(max_to_keep = 0)

    def train(self):
        g_optim = tf.train.AdamOptimizer(self.flags.lr, beta1 = self.flags.beta1) \
            .minimize(self.g_loss, var_list = self.g_vars)
        d1_optim = tf.train.AdamOptimizer(self.flags.lr, beta1 = self.flags.beta1) \
            .minimize(self.d1_loss, var_list = self.d1_vars)

        tf.global_variables_initializer().run()

        #merge summary
        g_sum_total = tf.summary.merge([self.g_loss_sum, self.im_sum, self.im_hat_sum])
        d1_sum_total = tf.summary.merge([self.d1_loss_sum, self.d1_fake_loss_sum, self.d1_real_loss_sum])

        writer = tf.summary.FileWriter("%s/dc_GAN_log_%s"%(self.flags.checkpoint_dir, self.flags.dataset_name), self.sess.graph)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord = coord )

        could_load, checkpoint_counter = load(self.saver, self.sess, self.flags)
        if could_load:
            counter = checkpoint_counter
            print(" [*] Load SUCCESS")
        else:
            counter = 0
            print(" [!] Load failed...")

        for i in xrange(counter, self.flags.iter):
            i += 1
            # run d1_optim
            _, sum_total_, d1_loss_ = self.sess.run([d1_optim, d1_sum_total, self.d1_loss])
            writer.add_summary(sum_total_, i)

            # run g_optim
            self.sess.run(g_optim)

            # run g_optim twice make sure d1_loss does not go to zero
            _, sum_total_, g_loss_ = self.sess.run([g_optim, g_sum_total, self.g_loss])
            writer.add_summary(sum_total_, i)

            print("iteration: [%2d], g_loss: %.8f, d1_loss: %.8f" % (i, g_loss_, d1_loss_))
            print('**************************')

            if np.mod(i,self.flags.save_iter) == 0 or i == self.flags.iter:
                # try to sample and save model
                [gt_im, sample_im] = self.sess.run([self.im, self.im_hat])
                save_images(self.flags, gt_im, i, 'GT')
                save_images(self.flags, sample_im, i, 'sample')

                save(self.saver, self.sess, self.flags, i)
                print ('saved once ...')
