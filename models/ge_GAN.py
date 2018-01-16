from __future__ import print_function
import os

import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from src.utils import setup_updates, save, load, save_images, get_training_data
from src.ops import KL_Gaussian, match, make_z
from src.net import net_G_32, net_E_32, net_G_64, net_E_64

class ge_GAN(object):
    def __init__ (self, sess, flags ):
        self.sess = sess
        self.flags = flags
        self.updates = setup_updates(flags)

        if self.flags.input_size == 32:
            self.generator = net_G_32(self.flags)
            self.encoder = net_E_32(self.flags)
        elif self.flags.input_size == 64:
            self.generator = net_G_64(self.flags)
            self.encoder = net_E_64(self.flags)
        else :
            print ('wrong input size: ', self.flags.input_size)
            raise

        self.batch_images = get_training_data(self.flags)
        self.build_model()

    def build_model(self):
        self.im = self.batch_images
        self.z = make_z(self.flags)

        self.im_sum = tf.summary.image('real_img', self.im)

        self.im_hat = self.generator.net(self.z)
        self.z_hat = self.encoder.net(self.im)
        # self.z_hat_test = self.encoder.net(self.im, reuse = True, is_training = False)
        self.z_hat_test = self.encoder.net(self.im, reuse = True)

        self.im_hat_sum = tf.summary.image('fake_img', self.im_hat)

        self.recon_im = self.generator.net(self.z_hat, reuse = True)
        # self.recon_im_test = self.generator.net(self.z_hat_test, reuse = True, is_training = False)
        self.recon_im_test = self.generator.net(self.z_hat_test, reuse = True)
        self.recon_z = self.encoder.net(self.im_hat, reuse = True)

        self.recon_im_sum = tf.summary.image('recon_im', self.recon_im)

        # generator loss

        # minizie the reccon_z with z in form of KL
        self.KL_fake_g_loss = KL_Gaussian(self.recon_z, direction = self.flags.KL)
        self.KL_fake_g_loss_sum = tf.summary.scalar('KL_fake_g_loss', self.KL_fake_g_loss)

        self.g_loss = self.KL_fake_g_loss * self.updates['g']['KL_fake']
        # if need to proceed reconstruction
        if self.updates['g']['match_z'] != 0:
            self.match_z_g_loss = match(self.recon_z, self.z, self.flags.match_z)
            self.match_z_g_loss_sum = tf.summary.scalar('match_z_g_loss', self.match_z_g_loss)
            self.g_loss += self.match_z_g_loss * self.updates['g']['match_z']
        if self.updates['g']['match_x'] != 0:
            self.match_x_g_loss = match(self.recon_im, self.im, self.flags.match_x)
            self.match_x_g_loss_sum = tf.summary.scalar('match_x_g_loss', self.match_x_g_loss)
            self.g_loss += self.match_x_g_loss * self.updates['g']['match_x']
        self.g_loss_sum = tf.summary.scalar('g_loss', self.g_loss)


        # encoder loss

        # minizie the z_hat with z in form of KL
        self.KL_real_e_loss = KL_Gaussian(self.z_hat, direction = self.flags.KL)
        self.KL_real_e_loss_sum = tf.summary.scalar('KL_real_e_loss', self.KL_real_e_loss)
        # maximize the reconz with z in form of KL
        self.KL_fake_e_loss = KL_Gaussian(self.recon_z, direction = self.flags.KL, is_minimize = False )
        self.KL_fake_e_loss_sum = tf.summary.scalar('KL_fake_e_loss', self.KL_fake_e_loss)
        self.e_loss = self.KL_fake_e_loss * self.updates['e']['KL_fake'] + self.KL_real_e_loss * self.updates['e']['KL_real']
        # if need to proceed reconstruction
        if self.updates['e']['match_z'] != 0:
            self.match_z_e_loss = match(self.recon_z, self.z, self.flags.match_z)
            self.match_z_e_loss_sum = tf.summary.scalar('match_z_e_loss', self.match_z_e_loss)
            self.e_loss += self.match_z_e_loss * self.updates['e']['match_z']
        if self.updates['e']['match_x'] != 0:
            self.match_x_e_loss = match(self.recon_im, self.im, self.flags.match_x)
            self.match_x_e_loss_sum = tf.summary.scalar('match_x_e_loss', self.match_x_e_loss)
            self.e_loss += self.match_x_e_loss * self.updates['e']['match_x']
        self.e_loss_sum = tf.summary.scalar('e_loss', self.e_loss)


        trainable_vars = tf.trainable_variables()
        self.g_vars = [var for var in trainable_vars if 'g_' in var.name]
        self.e_vars = [var for var in trainable_vars if 'e_' in var.name]

        self.saver = tf.train.Saver(max_to_keep = 0)

    def train(self):
        extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(extra_update_ops):
            g_optim = tf.train.AdamOptimizer(self.flags.lr, beta1 = self.flags.beta1) \
                .minimize(self.g_loss, var_list = self.g_vars)
            e_optim = tf.train.AdamOptimizer(self.flags.lr, beta1 = self.flags.beta1) \
                .minimize(self.e_loss, var_list = self.e_vars)

            # g_optim = tf.train.AdagradOptimizer(self.flags.lr).minimize(self.g_loss, var_list = self.g_vars)
            # e_optim = tf.train.AdagradOptimizer(self.flags.lr).minimize(self.e_loss, var_list = self.e_vars)


        tf.global_variables_initializer().run()

        # merge summary
        sum_total = tf.summary.merge([self.im_sum, self.im_hat_sum, self.recon_im_sum,
            self.g_loss_sum, self.KL_fake_g_loss_sum,
            self.e_loss_sum,self.KL_fake_e_loss_sum, self.KL_real_e_loss_sum])

        if hasattr(self, 'match_x_e_loss_sum'):
            sum_total = tf.summary.merge([sum_total, self.match_x_e_loss_sum])

        if hasattr(self, 'match_z_e_loss_sum'):
            sum_total = tf.summary.merge([sum_total, self.match_z_e_loss_sum])

        if hasattr(self, 'match_z_g_loss_sum'):
            sum_total = tf.summary.merge([sum_total, self.match_z_g_loss_sum])

        if hasattr(self, 'match_x_g_loss_sum'):
            sum_total = tf.summary.merge([sum_total, self.match_x_g_loss_sum])


        writer = tf.summary.FileWriter("%s/ge_GAN_log_%s"%(self.flags.checkpoint_dir, self.flags.dataset_name), self.sess.graph)

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
            for iter_e in range(self.updates['e']['num_updates']):
                if iter_e < (self.updates['e']['num_updates'] -1 ):
                    self.sess.run([e_optim])
                else:
                    # run with loss
                    # self.sess.run([e_optim])
                    _, e_loss_, KL_real_e_loss_, KL_fake_e_loss_ = self.sess.run(
                                        [e_optim, self.e_loss, self.KL_real_e_loss, self.KL_fake_e_loss])

            for iter_g in range(self.updates['g']['num_updates']):
                if iter_g < (self.updates['g']['num_updates'] -1):
                    self.sess.run([g_optim])
                else :
                    # run with summary and loss
                    # self.sess.run([g_optim])
                    _, sum_total_, g_loss_, KL_fake_g_loss_ = self.sess.run(
                                [g_optim, sum_total, self.g_loss, self.KL_fake_g_loss])
            writer.add_summary(sum_total_, i)

            print("iteration: [%2d], g_loss: %.8f, e_loss: %.8f" % (i, g_loss_, e_loss_))
            print ('fake/real_ e: ', KL_fake_e_loss_, '\\', KL_real_e_loss_)
            print ('fake_g: ', KL_fake_g_loss_)
            print('**************************')

            if np.mod(i,self.flags.save_iter) == 0 or i == self.flags.iter:
                # try to sample and save model
                [gt_im, recon_im] = self.sess.run([self.im, self.recon_im_test])
                save_images(self.flags, gt_im, i, 'GT')
                save_images(self.flags, recon_im, i, 'recon')

                save(self.saver, self.sess, self.flags, i)
                print ('saved once ...')
