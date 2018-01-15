from __future__ import print_function
import numpy as np
import tensorflow as tf
from src.ops import KL_Gaussian, match, make_z, normalize
import os

class Flags(object):
    def __init__(self):
        self.noise = 'sphere'
        self.batch_size = 2
        self.nz = 10

flags = Flags()
sess = tf.Session()
z = tf.convert_to_tensor(np.arange(20).reshape(2,10), dtype = tf.float16)
y = tf.convert_to_tensor(np.ones(20).reshape(2,10), dtype = tf.float16)
print (z)
print (sess.run(z))
after_normalized = normalize(z)
print (after_normalized)
print (sess.run(after_normalized))

match_L1 = match(z, y, dist = 'L1')
match_L2 = match(z, y, dist = 'L2')
match_cos = match(z, y, dist = 'cos')

print ('L1: ', sess.run(match_L1))
print ('L2: ', sess.run(match_L2))
print ('cos: ', sess.run(match_cos))

KL_v = KL_Gaussian(after_normalized, 'qp', is_minimize = False)
print (sess.run(KL_v))
