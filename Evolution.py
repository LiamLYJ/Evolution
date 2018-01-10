from __future__ import print_function
import os

import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from src.utils import *
from src.ops import *
from src.net import *

class Evo_GAN(obeject):
    def __init__ (self, sess, flags ):
        self.sess = sess
        self.flags = flags
        self.updates = setup_updates(flags)

        self.generator = net_G(self.flags)
        self.encoder = net_E(self.flags)
        self.batch_images = get_training_data(self.flags)
        self.build_model()

    def build_model(self):

        KL_Gaussian(samples, direction, is_minimize = True)

        self.im = self.batch_images
        self.z = make_z(self.flags)

        self.im_sum = tf.summary.image('real_img', self.im)

        self.im_hat = self.generator.net(self.z)
        self.z_hat = self.encoder.net(self.im)

        self.im_hat_sum = tf.summary.image('fake_img', self.im_hat)

        self.recon_im = self.encoder.net(self.z_hat)
        self.recon_z = self.generator.net(self.im_hat)

        self.recon_im = tf.summary.image('recon_im', self.recon_im)

        # generator loss

        # minizie the reccon_z with z in form of KL
        self.KL_fake_g_loss = KL_Gaussian(self.recon_z, direction = self.flags.KL)
        self.KL_fake_g_loss_sum = tf.summary.scalar('KL_fake_g_loss', self.KL_fake_g_loss)

        self.g_loss = self.KL_fake_g_loss * updates['g']['KL_fake']
        # if need to proceed reconstruction
        if updates['g']['match_z'] != 0:
            self.match_z_g_loss = match(self.recon_z, self.z, self.flags.match_z)
            self.match_z_g_loss_sum = tf.summary.scalar('match_z_g_loss', self.match_z_g_loss)
            self.g_loss += self.match_z_g_loss * updates['g']['match_z']
        if updates['g']['match_x'] != 0:
            self.match_x_g_loss = match(self.recon_im, self.im, self.flags.match_x)
            self.match_x_g_loss_sum = tf.summary.scalar('match_x_g_loss', self.match_x_g_loss)
            self.g_loss += self.match_x_g_loss * updates['g']['match_x']
        self.g_loss_sum = tf.summary.scalar('g_loss', self.g_loss)


        # encoder loss

        # minizie the z_hat with z in form of KL
        self.KL_real_e_loss = KL_Gaussian(self.z_hat, direction = self.flags.KL)
        self.KL_real_e_loss_sum = tf.summary.scalar('KL_real_e_loss', self.KL_real_e_loss)
        # maximize the reconz with z in form of KL
        self.KL_fake_e_loss = KL_Gaussian(self.recon_z, direction = self.flags.KL, is_minimize = False )
        self.KL_fake_e_loss_sum = tf.summary.scalar('KL_fake_e_loss', self.KL_fake_e_loss)
        self.e_loss = self.KL_fake_e_loss * updates['e']['KL_fake'] + self.KL_real_e_loss * updates['e']['KL_real']
        # if need to proceed reconstruction
        if updates['e']['match_z'] != 0:
            self.match_z_e_loss = match(self.recon_z, self.z, self.flags.match_z)
            self.match_z_e_loss_sum = tf.summary.scalar('match_z_e_loss', self.match_z_e_loss)
            self.e_loss += self.match_z_e_loss * updates['e']['match_z']
        if updates['e']['match_x'] != 0:
            self.match_x_e_loss = match(self.recon_im, self.im, self.flags.match_x)
            self.match_x_e_loss_sum = tf.summary.scalar('match_x_e_loss', self.match_x_e_loss)
            self.e_loss += self.match_x_e_loss * updates['e']['match_x']
        self.e_loss_sum = tf.summary.scalar('e_loss', self.e_loss)


        trainable_vars = tf.trainable_variables()
        self.g_vars = [var for var in trainable_vars if 'g_' in var.name]
        self.e_vars = [var for var in trainable_vars if 'e_' in var.name]

        self.saver = tf.train.Saver(max_to_keep = 0)

    def train(self):
        g_optim = tf.train.AdamOptimizer(self.flags.lr, beta1 = self.flags.beta1) \
            .minimize(self.loss_g, var_list = self.g_vars)
        e_optim = tf.train.AdamOptimizer(self.flags.lr, beta1 = self.flags.beta1) \
            .minimize(self.loss_e, var_list = self.e_vars)

        tf.global_variables_initializer().run()

        # merge summary
        self.sum_total = tf.summary.merge([self.im_sum, self.im_hat_sum, self.recon_im_sum,
            self.g_loss_sum, self.match_x_g_loss_sum, self.match_z_g_loss_sum, self.KL_fake_g_loss_sum,
            self.e_loss_sum, self.match_x_e_loss_sum, self.match_z_e_loss_sum, self.KL_fake_e_loss_sum, self.KL_real_e_loss_sum])

        self.writer = tf.summary.FileWriter("%s/Evo_GAN_log_%s"%(self.flags.checkpoint_dir, self.dataset_name), self.sess.graph)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord = coord )

        could_load, checkpoint_counter = self.load(self.checkpoint_dir)
        if could_load:
            counter = checkpoint_counter
            print(" [*] Load SUCCESS")
        else:
            counter = 0
            print(" [!] Load failed...")

        for i in xrange(counter, self.flags.iter):
            i += 1
            for iter_e in range(updates['e']['num_updates']):
                if iter_e < (updates['e']['num_updates'] -1 ):
                    self.sess.run([e_optim])
                else:
                    # run with loss
                    _, e_loss_ = self.sess.run([e_optim, self.e_loss])

            for iter_g in range(updates['g']['num_updates']):
                if iter_g < (updates['g']['num_updates'] -1):
                    self.sess.run([g_optim])
                else :
                    # run with summary and loss
                    _, sum_total_, g_loss_, = self.sess.run([g_optim, self.sum_total, self.g_loss])
            self.writer.add_summary(sum_total_, i)

            print("iteration: [%2d], g_loss: %.8f, e_loss: %.8f" % (i, g_loss_, e_loss_))
            print('**************************')

            if np.mod(i,self.flags.save_iter) == 0 or i == self.flags.iter:
                # try to sample and save model
                [gt_img, recon_img] = self.sess.run([self.im, self.recon_img])
                save_images(self.flags, gt_img, i, 'GT')
                save_images(self.flags, recon_img, i, 'recon')

                self.save(self.flags.checkpoint_dir, i)


flags = tf.app.flags
flags.DEFINE_integer("iter", 800000, "total iter to train ")
flags.DEFINE_integer("save_iter", 2000, "iter to save ")
flags.DEFINE_float("lr", 0.0002, "Learning rate of for adam [0.0002]")
flags.DEFINE_float("beta1", 0.5, "Momentum term of adam [0.5]")
flags.DEFINE_integer("batch_size", 64, "The size of batch images [64]")
flags.DEFINE_integer("input_height", 32, "The size of image to use (will be center cropped). [108]")
flags.DEFINE_integer("input_width", 32, "The size of image to use (will be center cropped). If None, same value as input_height [None]")
flags.DEFINE_string("dataset_name", "cifar10", "The name of dataset [...]")
flags.DEFINE_string("input_fname_pattern", '*jpg', "pattern of image ")
flags.DEFINE_string("checkpoint_dir", "./checkpoint_Evo_GAN", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("sample_dir", "./Evo_GAN_samples", "Directory name to save the image samples [samples]")
flags.DEFINE_boolean("train", True, "True for training, False for testing [False]")

flags.DEFINE_integer("nz",100, "length of lantent vector")
flags.DEFINE_integer("nc",3, " number of channel")
flags.DEFINE_integer("ngf", 64, "length of feature in Generator")
flags.DEFINE_integer("ndf", 64, "length of feature in Encoder")
flags.DEFINE_string("KL", "qp", "set which distribution to be unit gaussian| qp||pq")
flags.DEFINE_string("noise", "sphere", "kind of noise namely what kind of space to project on ")

# no need for reconstruction
flags.DEFINE_string("match_z", "L2", "loss type | L1|L2|cos")
flags.DEFINE_string("match_x", "L1", "loss type | L1|L2|cos")

flags.DEFINE_string("e_updates", "1;KL_fake:1,KL_real:1,match_z:0,match_x:0", "update plan and weights for encoder " )
flags.DEFINE_string("g_updates", "2;KL_fake:1,match_z:1,match_x:0", "update plan and weights for generator")


FLAGS = flags.FLAGS

def main(_):
    if not os.path.exists(FLAGS.checkpoint_dir):
        os.makedirs(FLAGS.checkpoint_dir)
    if not os.path.exists(FLAGS.sample_dir):
        os.makedirs(FLAGS.sample_dir)
    if not os.path.exists(os.path.join(FLAGS.sample_dir, FLAGS.dataset_name)):
        os.makedirs(os.path.join(FLAGS.sample_dir, FLAGS.dataset_name))

    run_config = tf.ConfigProto()
    run_config.gpu_options.allow_growth=True

    with tf.Session(config = run_config) as sess:
        gan = Evo_GAN(sess, FLAGS)
        gan.train()

if __name__ == '__main__':
    tf.app.run()
