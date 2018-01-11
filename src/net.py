from __future__ import print_function
import tensorflow as tf
# from ops import *
import tensorflow.contrib.slim as slim

def model_arg_scope(weight_decay=0.0005, is_training = True, ac_fn = tf.nn.relu):
    batch_norm_params = {
        'is_training': is_training,
        'center': True,
        'scale': True,
        'decay': 0.9997,
        'epsilon': 0.001,
    }
    with slim.arg_scope([slim.conv2d, slim.fully_connected,slim.conv2d_transpose],
    activation_fn = ac_fn,
    weights_regularizer= slim.l2_regularizer(weight_decay),
    weights_initializer= tf.truncated_normal_initializer(stddev=0.01),
    biases_initializer= tf.zeros_initializer(),
    normalizer_fn = slim.batch_norm
    ):
        with slim.arg_scope([slim.batch_norm], **batch_norm_params):
            with slim.arg_scope([slim.conv2d,slim.conv2d_transpose], padding = 'SAME') as sc:
                return sc


def _leaky_relu(x):
    return tf.where(tf.greater(x,0),x,0.2*x)

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
                net = slim.fully_connected(input, 4*4*self.ngf*8, scope = 'g_fc')
                # b *4*4 *
                net = tf.reshape(net, [-1, 4,4, self.ngf*8])
                # b* 4*4 *
                net = slim.conv2d_transpose(net, self.ngf*8, [5,5], stride = 1, scope = 'g_deconv_1')
                # b* 8*8 *
                net = slim.conv2d_transpose(net, self.ngf*4, [5,5], stride = 2 , scope = 'g_deconv_2')
                # b* 16*16 *
                net = slim.conv2d_transpose(net, self.ngf*2, [5,5], stride = 2 , scope = 'g_deconv_3')
                # b* 32*32 *
                net = slim.conv2d_transpose(net, self.ngf, [5,5], stride = 2 , scope = 'g_deconv_4')

                net = slim.conv2d(net, self.nc, [1,1],stride = 1, activation_fn = tf.nn.tanh, scope = 'g_conv')

                return net

class net_E(object):
    def __init__(self,flags):
        self.nc = flags.nc
        self.nz = flags.nz
        self.nef = flags.nef
    def net(self, input, is_training = True, reuse = False, scope = 'Encoder'):
        with tf.variable_scope(scope, 'Encoder', [input], reuse = reuse ) as net_scope:
            with slim.arg_scope(model_arg_scope(is_training = is_training, ac_fn = _leaky_relu)):
                if reuse:
                    net_scope.reuse_variables()
                net = slim.conv2d(input, self.nef, [5,5], stride = 2, normalizer_fn = None, scope = 'e_conv_1')
                net = slim.conv2d(net, self.nef*2, [5,5], stride = 2, scope = 'e_conv_2')
                net = slim.conv2d(net, self.nef*4, [5,5], stride = 2, scope = 'e_conv_3')
                net = slim.conv2d(net, self.nz, [5,5], stride = 2, normalizer_fn = None, activation_fn = None, scope = 'e_conv_4')
                net = slim.avg_pool2d(net,[2,2], scope = 'e_pool')
                return net

class net_D1(object):
    def __init__(self,flags):
        self.nd1f = flags.nd1f

    def net(self, input, is_training = True, reuse = False, scope = 'Discriminator_1'):
        with tf.variable_scope(scope, 'Discriminator_1', [input], reuse = reuse) as net_scope:
            with slim.arg_scope(model_arg_scope(is_training = is_training, ac_fn = _leaky_relu )):
                if reuse:
                    net_scope.reuse_variables()
                net = slim.conv2d(input, self.nd1f, [5,5], stride =2, scope = 'd1_conv1')
                net = slim.conv2d(net, self.nd1f * 2, [5,5], stride =2, scope = 'd1_conv2')
                net = slim.conv2d(net, self.nd1f * 4, [5,5], stride =2, scope = 'd1_conv3')
                net = slim.conv2d(net, self.nd1f * 8, [5,5], stride =2, scope = 'd1_conv4')
                net = slim.flatten(net)
                net = slim.fully_connected(net, 1, scope = 'd1_fc', activation_fn = None, normalizer_fn = None)
                return net


# class net_D2(object):
#     def __init__(self,flags):
