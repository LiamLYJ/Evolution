from __future__ import print_function
import os

import numpy as np
import tensorflow as tf

from models.ge_GAN import ge_GAN
from models.dc_GAN import dc_GAN
from models.evo_GAN import evo_GAN

flags = tf.app.flags
flags.DEFINE_integer("iter", 800000, "total iter to train ")
flags.DEFINE_integer("save_iter", 2000, "iter to save ")
flags.DEFINE_float("lr", 0.0001, "Learning rate of for adam [0.0002]")
flags.DEFINE_float("beta1", 0.5, "Momentum term of adam [0.5]")
flags.DEFINE_integer("batch_size", 64, "The size of batch images [64]")
flags.DEFINE_integer("input_height", 32, "The size of image to use (will be center cropped). [108]")
flags.DEFINE_integer("input_width", 32, "The size of image to use (will be center cropped). If None, same value as input_height [None]")
flags.DEFINE_string("dataset_name", "cifar10", "The name of dataset [...]")
flags.DEFINE_string("input_fname_pattern", '*jpg', "pattern of image ")
flags.DEFINE_string("model_name", 'ge_GAN', 'model to use ')

flags.DEFINE_boolean("train", True, "True for training, False for testing [False]")

flags.DEFINE_integer("nz",100, "length of lantent vector")
flags.DEFINE_integer("nc",3, " number of channel")
flags.DEFINE_integer("ngf", 64, "length of feature in Generator")
flags.DEFINE_integer("nef", 64, "length of feature in Encoder")
flags.DEFINE_integer("nd1f", 64, "length of feature in Discriminator_1")
flags.DEFINE_string("KL", "qp", "set which distribution to be unit gaussian| qp||pq")
flags.DEFINE_string("noise", "sphere", "kind of noise namely what kind of space to project on ")

# no need for reconstruction
flags.DEFINE_string("match_z", "L2", "loss type | L1|L2|cos")
flags.DEFINE_string("match_x", "L1", "loss type | L1|L2|cos")

flags.DEFINE_string("e_updates", "1;KL_fake:1,KL_real:1,match_z:0,match_x:0", "update plan and weights for encoder " )
flags.DEFINE_string("g_updates", "2;KL_fake:1,match_z:1,match_x:0", "update plan and weights for generator")


FLAGS = flags.FLAGS

FLAGS.checkpoint_dir = "./checkpoint_%s"%FLAGS.model_name
FLAGS.sample_dir = "./%s_samples"%FLAGS.model_name

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
        if FLAGS.model_name == 'ge_GAN':
            gan = ge_GAN(sess, FLAGS)
            gan.train()
        elif FLAGS.model_name == 'dc_GAN':
            gan = dc_GAN(sess, FLAGS)
            gan.train()
        elif FLAGS.model_name == 'evo_GAN':
            gan = evo_GAN(sess, FLAGS)
            gan.train()
        else :
            print ('FLAGS.model_name is illega model type ! ')

if __name__ == '__main__':
    tf.app.run()
