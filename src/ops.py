from __future__ import print_function
import tensorflow as tf

def make_z(flags):
    z = tf.random_normal([flags.batch_size, 1,1, flags.nz])
    if flags.noise == 'sphere':
        normalize(z)
    return z

def normalize(x, dim = 1):
    # project into sphere
    x = tf.squeeze(x)
    tmp = tf.div(x,tf.expand_dims(tf.norm(x,axis = 1),1))
    # expand_dims to [b *1 * 1 *nz]
    return tf.expand_dims(tf.expand_dims(tmp, axis= 1), axis = 1)

def match(x,y, dist):
    if dist == 'L2':
        return tf.losses.mean_squared_error(x,y)
    elif dist == 'L1':
        return tf.losses.absolute_difference(x,y)
    elif dist == 'cos':
        x_normalized = normalize(x)
        y_normalized = normalize(y)
        return 2 - tf.reduce_mean(tf.multiply(x_normalized, y_normalized))
    else :
        print ("None loss indicated ,, wrong ")
        raise

def KL_Gaussian(samples, direction,is_minimize = True):

    assert direction in ['pq','qp']
    samples_mean, samples_var = tf.nn.moments(tf.squeeze(samples), axes = [1] )
    if direction == 'pq':
        # mu_1 = 0; sigma_1 = 1
        term1 = (1+ tf.pow(samples_mean, 2)) / (2 * tf.pow(samples_var, 2))
        term2 = tf.log(samples_var)
        KL = tf.reduce_mean(term1 + term2 -0.5)
    else :
        # mu_2 = 0; sigma_2 = 1
        term1 = (tf.pow(samples_mean, 2) + tf.pow(samples_var, 2)) / 2
        term2 = -tf.log(samples_var)
        KL = tf.reduce_mean(term1 + term2 - 0.5)

    if not is_minimize:
        KL *= -1

    return KL
