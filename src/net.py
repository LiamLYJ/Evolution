from __future__ import print_function
import tensorflow as tf
from ops import *
import tensorflow.contrib.slim as slim

def model_arg_scope(self, weight_decay=0.0005, is_training = True):
    batch_norm_params = {
        'is_training': is_training,
        'center': True,
        'scale': True,
        'decay': 0.9997,
        'epsilon': 0.001,
    }
    with slim.arg_scope([slim.conv2d, slim.fully_connected,slim.conv2d_transpose],
    activation_fn = tf.nn.relu,
    weights_regularizer= slim.l2_regularizer(weight_decay),
    weights_initializer= tf.truncated_normal_initializer(stddev=0.01),
    biases_initializer= tf.zeros_initializer(),
    normalizer_fn = slim.batch_norm
    ):
        with slim.arg_scope([slim.batch_norm], **batch_norm_params):
            with slim.arg_scope([slim.conv2d,slim.conv2d_transpose], padding = 'SAME') as sc:
                return sc

class net_G(object):
    def __init__(self,flags):
        self.nc = flags.nc
        self.nz = flags.nz
        self.ngf = flags.ngf

    def net(self, input, is_training = True, reuse = False, scope = 'Generator'):
        with tf.variable_scope(scope, 'Generator', [input], reuse = reuse ) as net_scope:
            with slim.arg_scope(model_arg_scope(is_training = is_training)):
                if reuse:
                    net_scope.reuse_variables()
                net = slim.conv2d_transpose(input, self.ngf*8, [4,4], stride = 1, scope = 'g_deconv_1')
                net = slim.conv2d_transpose(net, self.ngf*4, [4,4], stride = 2 , scope = 'g_deconv_2')
                net = slim.conv2d_transpose(net, self.ngf*2, [4,4], stride = 2 , scope = 'g_deconv_3')
                net = slim.conv2d_transpose(net, self.ngf*2, [4,4], stride = 2 , scope = 'g_deconv_4')

                net = slim.conv2d(net, self.nc, [1,1],stride = 1, activation_fn = tf.nn.tanh, scope = 'g_conv')
                return net

class net_E(object):
    def __init__(self,flags):
        self.nc = flags.nc
        self.nz = flags.nz
        self.ndf = flags.ndf
    def net(self, input, is_training = True, reuse = False, scope = 'Encoder'):
        with tf.variable_scope(scope, 'Encoder', [input], reuse = reuse ) as net_scope:
            with slim.arg_scope(model_arg_scope(is_training = is_training), activation_fn = tf.nn.leaky_relu):
                if reuse:
                    net_scope.reuse_variables()
                net = slim.conv2d(input, ndf, [4,4], stride = 2, normalizer_fn = None, scope = 'e_conv_1')
                net = slim.conv2d(net, ndf*2, [4,4], stride = 2, scope = 'e_conv_2')
                net = slim.conv2d(net, ndf*4, [4,4], stride = 2, scope = 'e_conv_3')
                net = slim.conv2d(net, nz, [4,4], stride = 2, normalizer_fn = None, activation_fn = None, scope = 'e_conv_3')
                net = slim.avg_pool2d(net,[2,2], scope = 'e_pool')

                return net
