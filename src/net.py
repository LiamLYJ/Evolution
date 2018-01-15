from __future__ import print_function
import tensorflow as tf
from ops import normalize
import tensorflow.contrib.slim as slim

def model_arg_scope(weight_decay=0.0005, is_training = True, ac_fn = tf.nn.relu):
    batch_norm_params = {
        'is_training': is_training,
        'center': True,
        'scale': True,
        'decay': 0.9997,
        'epsilon': 0.00001,
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

class net_G_32(object):
    def __init__(self,flags):
        self.nc = flags.nc
        self.nz = flags.nz
        self.ngf = flags.ngf

    def net(self, input, is_training = True, reuse = False, scope = 'Generator'):
        input = tf.squeeze(input)
        input = tf.expand_dims(tf.expand_dims(input, axis= 1), axis = 1)
        with tf.variable_scope(scope, 'Generator', [input], reuse = reuse ) as net_scope:
            with slim.arg_scope(model_arg_scope(is_training = is_training)):
                if reuse:
                    net_scope.reuse_variables()
                # b* 4*4
                net = slim.conv2d_transpose(input, self.ngf*8, [4,4], padding = 'VALID',stride = 1, scope = 'g_deconv_1')
                # net = slim.fully_connected(input, self.ngf*8*4*4, scope = 'g_deconv_1')
                # net = tf.reshape(net, [-1, 4, 4, self.ngf*8])
                # b* 8*8 *
                net = slim.conv2d_transpose(net, self.ngf*4, [5,5], stride = 2 , scope = 'g_deconv_2')
                # b* 16*16 *
                net = slim.conv2d_transpose(net, self.ngf*2, [5,5], stride = 2 , scope = 'g_deconv_3')
                # b* 32*32 *
                net = slim.conv2d_transpose(net, self.ngf*2, [5,5], stride = 2 , scope = 'g_deconv_4')

                net = slim.conv2d(net, self.nc, [1,1],stride = 1, activation_fn = tf.nn.tanh, scope = 'g_conv')

                return net


class net_G_64(object):
    def __init__(self,flags):
        self.nc = flags.nc
        self.nz = flags.nz
        self.ngf = flags.ngf

    def net(self, input, is_training = True, reuse = False, scope = 'Generator'):
        input = tf.squeeze(input)
        input = tf.expand_dims(tf.expand_dims(input, axis= 1), axis = 1)
        with tf.variable_scope(scope, 'Generator', [input], reuse = reuse ) as net_scope:
            with slim.arg_scope(model_arg_scope(is_training = is_training)):
                if reuse:
                    net_scope.reuse_variables()
                # b* 4*4
                net = slim.conv2d_transpose(input, self.ngf*8, [4,4], padding = 'VALID', stride = 1, scope = 'g_deconv_1')
                # net = slim.fully_connected(input, self.ngf*8*4*4, scope = 'g_deconv_1')
                # net = tf.reshape(net, [-1, 4, 4, self.ngf*8])
                # b* 8*8 *
                net = slim.conv2d_transpose(net, self.ngf*4, [5,5], stride = 2 , scope = 'g_deconv_2')
                # b* 16*16 *
                net = slim.conv2d_transpose(net, self.ngf*2, [5,5], stride = 2 , scope = 'g_deconv_3')
                # b* 32*32 *
                net = slim.conv2d_transpose(net, self.ngf, [5,5], stride = 2 , scope = 'g_deconv_4')
                # b* 64*64 *
                net = slim.conv2d_transpose(net, self.nc, [5,5], stride = 2 , activation_fn = tf.nn.tanh, scope = 'g_deconv_5')

                return net


class net_E_32(object):
    def __init__(self,flags):
        self.nc = flags.nc
        self.nz = flags.nz
        self.nef = flags.nef
        self.noise = flags.noise

    def net(self, input, is_training = True, reuse = False, scope = 'Encoder'):
        with tf.variable_scope(scope, 'Encoder', [input], reuse = reuse ) as net_scope:
            with slim.arg_scope(model_arg_scope(is_training = is_training, ac_fn = _leaky_relu)):
                if reuse:
                    net_scope.reuse_variables()
                # b*16*16
                net = slim.conv2d(input, self.nef, [5,5], stride = 2, normalizer_fn = None, scope = 'e_conv_1')
                # b*8*8
                net = slim.conv2d(net, self.nef*2, [5,5], stride = 2, scope = 'e_conv_2')
                # b*4*4
                net = slim.conv2d(net, self.nef*4, [5,5], stride = 2, scope = 'e_conv_3')
                # b*2*2
                net = slim.conv2d(net, self.nz, [5,5], stride = 2, normalizer_fn = None, activation_fn = None, scope = 'e_conv_4')
                # b*1*1
                net = slim.avg_pool2d(net,[2,2], scope = 'e_pool')

                if self.noise == 'sphere':
                    net = normalize(net)

                return net


class net_E_64(object):
    def __init__(self,flags):
        self.nc = flags.nc
        self.nz = flags.nz
        self.nef = flags.nef
        self.noise = flags.noise

    def net(self, input, is_training = True, reuse = False, scope = 'Encoder'):
        with tf.variable_scope(scope, 'Encoder', [input], reuse = reuse ) as net_scope:
            with slim.arg_scope(model_arg_scope(is_training = is_training, ac_fn = _leaky_relu)):
                if reuse:
                    net_scope.reuse_variables()
                #b*32*32
                net = slim.conv2d(input, self.nef, [5,5], stride = 2, normalizer_fn = None, scope = 'e_conv_1')
                #b*16*16
                net = slim.conv2d(net, self.nef*2, [5,5], stride = 2, scope = 'e_conv_2')
                #b*8*8
                net = slim.conv2d(net, self.nef*4, [5,5], stride = 2, scope = 'e_conv_3')
                #b*4*4
                net = slim.conv2d(net, self.nef*8, [5,5], stride = 2, scope = 'e_conv_4')
                #b*1*1
                net = slim.conv2d(net, self.nz, [4,4], stride = 1, padding = 'VALID', normalizer_fn = None, activation_fn = None, scope = 'e_conv_5')

                if self.noise == 'sphere':
                    net = normalize(net)

                return net


class net_D1_32(object):
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


class net_D1_64(object):
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
                net = slim.conv2d(net, self.nd1f * 8, [5,5], stride =2, scope = 'd1_conv5')
                net = slim.flatten(net)
                net = slim.fully_connected(net, 1, scope = 'd1_fc', activation_fn = None, normalizer_fn = None)
                return net


class net_D2_32(object):
    def __init__(self,flags):
        self.nd2f = max(flags.nd2f, 2* flags.nz)
        self.naim = self.nd2f - flags.nz
        self.input_size = flags.input_size

    def net(self, im_input, z_input, is_training = True, reuse = False, scope = 'Discriminator_2'):
        with tf.variable_scope(scope, 'Discriminator_2', [im_input, z_input], reuse = reuse) as net_scope:
            with slim.arg_scope(model_arg_scope(is_training = is_training, ac_fn = _leaky_relu )):
                if reuse:
                    net_scope.reuse_variables()
                im_flat = slim.flatten(im_input)
                im_flat = slim.fully_connected(im_flat, self.input_size*self.input_size, scope = 'd2_im_flat_fc1')
                im_flat = slim.fully_connected(im_flat, self.naim, scope = 'd2_im_flat_fc2')

                pair = tf.concat([im_flat,tf.squeeze(z_input)], 1)
                net = slim.fully_connected(pair, self.nd2f, scope = 'd2_fc1')
                net = slim.fully_connected(net, self.nd2f*2, scope = 'd2_fc2')
                net = slim.fully_connected(net, self.nd2f*4, scope = 'd2_fc3')

                net = slim.flatten(net)
                net = slim.fully_connected(net, 1, normalizer_fn = None, activation_fn = None, scope = 'd2_fc4')
                return net


class net_D2_64(object):
    def __init__(self,flags):
        self.nd2f = max(flags.nd2f, 2* flags.nz)
        self.naim = self.nd2f - flags.nz
        self.input_size = flags.input_size

    def net(self, im_input, z_input, is_training = True, reuse = False, scope = 'Discriminator_2'):
        with tf.variable_scope(scope, 'Discriminator_2', [im_input, z_input], reuse = reuse) as net_scope:
            with slim.arg_scope(model_arg_scope(is_training = is_training, ac_fn = _leaky_relu )):
                if reuse:
                    net_scope.reuse_variables()
                im_input = tf.image.resize_images(im_input,[self.input_size/2, self.input_size/2])
                im_flat = slim.flatten(im_input)
                im_flat = slim.fully_connected(im_flat, self.input_size*self.input_size, scope = 'd2_im_flat_fc1')
                im_flat = slim.fully_connected(im_flat, self.naim, scope = 'd2_im_flat_fc2')

                pair = tf.concat([im_flat,tf.squeeze(z_input)], 1)
                net = slim.fully_connected(pair, self.nd2f, scope = 'd2_fc1')
                net = slim.fully_connected(net, self.nd2f*2, scope = 'd2_fc2')
                net = slim.fully_connected(net, self.nd2f*4, scope = 'd2_fc3')

                net = slim.flatten(net)
                net = slim.fully_connected(net, 1, normalizer_fn = None, activation_fn = None, scope = 'd2_fc4')
                return net
