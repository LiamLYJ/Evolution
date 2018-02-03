from __future__ import print_function
import os

import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from src.utils import setup_updates, save, load, save_images, get_training_data
from src.ops import KL_Gaussian, make_z
from src.net import net_G_sr, net_E_sr, net_D1_sr, net_D2_sr

class sr_GAN(object):
    def __init__ (self, sess, flags ):
        self.sess = sess
        self.flags = flags
        self.batch_size = flags.batch_size
        self.updates = setup_updates(flags)

        self.generator = net_G_sr(self.flags)
        self.encoder = net_E_sr(self.flags)
        self.discriminator_1 = net_D1_sr(self.flags)
        self.discriminator_2 = net_D2_sr(self.flags)

        self.batch_images = get_training_data(self.flags)
        self.build_model()

    def build_model(self):
        self.imh = self.batch_images
        self.iml_tmp = tf.image.resize_images(self.imh, [8,8])
        self.iml = tf.image.resize_images(self.iml_tmp, [32,32])

        self.imh_sum = tf.summary.image('img_high', self.imh)
        self.iml_sum = tf.summary.image('img_low', self.iml)

        self.imh_hat = self.generator.net(self.iml)
        self.iml_hat = self.encoder.net(self.imh)

        self.imh_hat_sum = tf.summary.image('img_high_fake', self.imh_hat)

        # self.recon_im = self.generator.net(self.z_hat, reuse = True)
        # self.recon_z = self.encoder.net(self.im_hat, reuse = True)

        # self.recon_im_sum = tf.summary.image('recon_im', self.recon_im)

        self.d1_real = self.discriminator_1.net(self.imh)
        self.d1_fake = self.discriminator_1.net(self.imh_hat, reuse = True)

        self.d2_real = self.discriminator_2.net(self.imh, self.iml)
        self.d2_fake_imh_hat = self.discriminator_2.net(self.imh_hat, self.iml, reuse = True)
        self.d2_fake_iml_hat = self.discriminator_2.net(self.imh, self.iml_hat, reuse = True)


        # generator loss

        # # minizie the reccon_z with z in form of KL
        # self.KL_fake_g_loss = KL_Gaussian(self.recon_z, direction = self.flags.KL)
        # self.KL_fake_g_loss_sum = tf.summary.scalar('KL_fake_g_loss', self.KL_fake_g_loss)

        self.d1_g_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits = self.d1_fake, labels = tf.ones([self.batch_size], dtype = tf.int64)
        ))
        self.d1_g_loss_sum = tf.summary.scalar('d1_g_loss', self.d1_g_loss)
        self.d2_imh_hat_g_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits = self.d2_fake_imh_hat, labels = tf.ones([self.batch_size], dtype = tf.int64)
        ))
        self.d2_imh_hat_g_loss_sum = tf.summary.scalar('d2_imh_hat_g_loss', self.d2_imh_hat_g_loss)

        # self.g_loss = self.KL_fake_g_loss * self.updates['g']['KL_fake'] + \
        #                 self.d1_g_loss * self.updates['g']['d1_fake'] + \
        #                 self.d2_im_hat_g_loss * self.updates['g']['d2_fake']
        self.g_loss =   self.d1_g_loss * self.updates['g']['d1_fake'] + \
                        self.d2_imh_hat_g_loss * self.updates['g']['d2_fake']
        self.g_loss_sum = tf.summary.scalar('g_loss', self.g_loss)


        # encoder loss

        # # minizie the z_hat with z in form of KL
        # self.KL_real_e_loss = KL_Gaussian(self.z_hat, direction = self.flags.KL)
        # self.KL_real_e_loss_sum = tf.summary.scalar('KL_real_e_loss', self.KL_real_e_loss)
        # # maximize the reconz with z in form of KL
        # self.KL_fake_e_loss = KL_Gaussian(self.recon_z, direction = self.flags.KL, is_minimize = False )
        # self.KL_fake_e_loss_sum = tf.summary.scalar('KL_fake_e_loss', self.KL_fake_e_loss)

        self.d2_iml_hat_e_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits = self.d2_fake_iml_hat, labels = tf.ones([self.batch_size], dtype = tf.int64)
        ))
        self.d2_iml_hat_e_loss_sum = tf.summary.scalar('d2_iml_hat_e_loss', self.d2_iml_hat_e_loss)

        # self.e_loss = self.KL_fake_e_loss * self.updates['e']['KL_fake'] + \
        #                 self.KL_real_e_loss * self.updates['e']['KL_real'] + \
        #                 self.d2_z_hat_e_loss * self.updates['e']['d2_fake']
        self.e_loss =  self.d2_iml_hat_e_loss * self.updates['e']['d2_fake']
        self.e_loss_sum = tf.summary.scalar('e_loss', self.e_loss)


        # discriminator_1 loss

        self.d1_fake_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits = self.d1_fake, labels = tf.zeros([self.batch_size], dtype = tf.int64)
        ))
        self.d1_real_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits = self.d1_real, labels = tf.ones([self.batch_size], dtype = tf.int64)
        ))
        self.d1_loss = self.updates['d1']['whole_weight'] * \
                        (self.updates['d1']['fake_weight'] * self.d1_fake_loss + \
                        self.updates['d1']['real_weight'] * self.d1_real_loss ) /2
        self.d1_loss_sum = tf.summary.scalar('d1_loss', self.d1_loss)

        # discriminator_2 loss

        self.d2_real_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits = self.d2_real , labels = tf.ones([self.batch_size], dtype = tf.int64)
        ))
        self.d2_fake_iml_hat_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits = self.d2_fake_iml_hat, labels = tf.zeros([self.batch_size], dtype = tf.int64)
        ))
        self.d2_fake_imh_hat_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits = self.d2_fake_imh_hat, labels = tf.zeros([self.batch_size], dtype = tf.int64)
        ))
        self.d2_loss = self.updates['d2']['whole_weight'] * \
                        (self.updates['d2']['real_weight'] * self.d2_real_loss + \
                         self.updates['d2']['fake_im_weight'] * self.d2_fake_imh_hat_loss + \
                         self.updates['d2']['fake_z_weight'] * self.d2_fake_iml_hat_loss) /3
        self.d2_loss_sum = tf.summary.scalar('d2_loss', self.d2_loss)

        # get variables

        trainable_vars = tf.trainable_variables()
        self.g_vars = [var for var in trainable_vars if 'g_' in var.name]
        self.e_vars = [var for var in trainable_vars if 'e_' in var.name]
        self.d1_vars = [var for var in trainable_vars if 'd1_' in var.name]
        self.d2_vars = [var for var in trainable_vars if 'd2_' in var.name]

        self.saver = tf.train.Saver(max_to_keep = 0)

    def train(self):
        extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(extra_update_ops):
            g_optim = tf.train.AdamOptimizer(self.flags.lr, beta1 = self.flags.beta1) \
                .minimize(self.g_loss, var_list = self.g_vars)
            e_optim = tf.train.AdamOptimizer(self.flags.lr, beta1 = self.flags.beta1) \
                .minimize(self.e_loss, var_list = self.e_vars)
            # d1_optim = tf.train.AdamOptimizer(self.flags.lr, beta1 = self.flags.beta1) \
            #     .minimize(self.d1_loss, var_list = self.d1_vars)
            # d2_optim = tf.train.AdamOptimizer(self.flags.lr, beta1 = self.flags.beta1) \
            #     .minimize(self.d2_loss, var_list = self.d2_vars)
            d1_optim = tf.train.RMSPropOptimizer(self.flags.lr).minimize(self.d1_loss, var_list = self.d1_vars)
            d2_optim = tf.train.RMSPropOptimizer(self.flags.lr).minimize(self.d2_loss, var_list = self.d2_vars)
        tf.global_variables_initializer().run()

        # merge summary
        # sum_total = tf.summary.merge([self.im_sum, self.im_hat_sum, self.recon_im_sum,
        #     self.g_loss_sum, self.KL_fake_g_loss_sum, self.d1_g_loss_sum, self.d2_im_hat_g_loss_sum,
        #     self.e_loss_sum,self.KL_fake_e_loss_sum, self.KL_real_e_loss_sum, self.d2_z_hat_e_loss_sum,
        #     self.d1_loss_sum, self.d2_loss_sum
        #     ])
        sum_total = tf.summary.merge([self.imh_sum, self.imh_hat_sum,
            self.g_loss_sum, self.d1_g_loss_sum, self.d2_imh_hat_g_loss_sum,
            self.e_loss_sum, self.d2_iml_hat_e_loss_sum,
            self.d1_loss_sum, self.d2_loss_sum
            ])

        writer = tf.summary.FileWriter("%s/sr_GAN_log_%s"%(self.flags.checkpoint_dir, self.flags.dataset_name), self.sess.graph)

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

            _,_, d1_loss_, d2_loss_ = self.sess.run([d1_optim, d2_optim, self.d1_loss, self.d2_loss])

            for iter_e in range(self.updates['e']['num_updates']):
                if iter_e < (self.updates['e']['num_updates'] -1 ):
                    self.sess.run([e_optim])
                else:
                    # run with loss
                    _, e_loss_ = self.sess.run([e_optim, self.e_loss])

            for iter_g in range(self.updates['g']['num_updates']):
                if iter_g < (self.updates['g']['num_updates'] -1):
                    self.sess.run([g_optim])
                else :
                    # run with summary and loss
                    _, sum_total_, g_loss_, = self.sess.run([g_optim, sum_total, self.g_loss])

            if np.mod(i,10) == 0:
                writer.add_summary(sum_total_, i)
                print("iteration: [%2d], g_loss: %.8f, e_loss: %.8f, d1_loss: %.8f, d2_loss: %.8f" \
                        % (i, g_loss_, e_loss_, d1_loss_, d2_loss_ ))
                print('**************************')

            if np.mod(i,self.flags.save_iter) == 0 or i == self.flags.iter:
                # try to sample and save model
                [low_im, high_im, fakehigh_im] = self.sess.run([self.iml, self.imh, self.imh_hat])
                save_images(self.flags, low_im, i, 'iml')
                save_images(self.flags, high_im, i, 'imh')
                save_images(self.flags, fakehigh_im, i, 'imh_hat')

                save(self.saver, self.sess, self.flags, i)
                print ('saved once ...')
