from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import os
import numpy as np
from PIL import Image
import cv2
from tensorflow.examples.tutorials.mnist import input_data
from utils import *
import sys
from six.moves import urllib
import tarfile



class Cifar10Record(object):
    width = 32
    height = 32
    depth = 3

    def set_label(self, label_byte):
        self.label = np.frombuffer(label_byte, dtype=np.uint8)

    def set_image(self, image_bytes):
        byte_buffer = np.frombuffer(image_bytes, dtype=np.int8)
        reshaped_array = np.reshape(byte_buffer,
                                [self.depth, self.width, self.height])
        self.byte_array = np.transpose(reshaped_array, [1, 2, 0])
        self.byte_array = self.byte_array.astype(np.float32)

class Cifar10Reader(object):
    def __init__(self, filename):
        if not os.path.exists(filename):
            print(filename + ' is not exist')
            return

        self.bytestream = open(filename, mode="rb")

    def close(self):
        if not self.bytestream:
            self.bytestream.close()

    def read(self, index):
        result = Cifar10Record()

        label_bytes = 1
        image_bytes = result.height * result.width * result.depth
        record_bytes = label_bytes + image_bytes

        self.bytestream.seek(record_bytes * index, 0)

        result.set_label(self.bytestream.read(label_bytes))
        result.set_image(self.bytestream.read(image_bytes))

        return result


FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('file', '/home/gpu_server/Downloads/cifar-10-batches-bin/', "path")
tf.app.flags.DEFINE_integer('offset', 0, "start index")
tf.app.flags.DEFINE_integer('length', 10000, "end index")

if not os.path.exists('./train_data'):
    os.mkdir('./train_data')

if not os.path.exists('./test_data'):
    os.mkdir('./test_data')

# for i in range(10):
#     if not os.path.exists('./train_data/%d'%i):
#         os.mkdir('./train_data/%d'%i)
#     if not os.path.exists('./test_data/%d'%i):
#         os.mkdir('./test_data/%d'%i)


if not os.path.exists('./train_data/cifar10'):
    os.mkdir('./train_data/cifar10')

if not os.path.exists('./test_data/cifar10'):
    os.mkdir('./test_data/cifar10')

basename = os.path.basename(FLAGS.file)
stop = FLAGS.offset + FLAGS.length
print ('start to convert training data')
# reader = Cifar10Reader(FLAGS.file)
for i in range(1,6):
    reader = Cifar10Reader(FLAGS.file + 'data_batch_%d.bin'%i)
    for index in range(FLAGS.offset, stop):
        image = reader.read(index)

        # print('label: %d' % image.label)
        # path = './train_data/%d/'%image.label
        path = './train_data/cifar10'
        imageshow = Image.fromarray(image.byte_array.astype(np.uint8))

        file_name = '%s-%02d-%d.jpg' % (basename, index, image.label)
        file = os.path.join(path, file_name)

        with open(file, mode='wb') as out:
            imageshow.save(out, format='JPEG')
        # print ('index is :',index)
    reader.close()
print ('start to convert testing data')
reader = Cifar10Reader(FLAGS.file+'test_batch.bin')
for index in range(FLAGS.offset, stop):
    image = reader.read(index)

    # print('label: %d' % image.label)
    # path = './test_data/%d/'%image.label
    path = './test_data/cifar10'
    imageshow = Image.fromarray(image.byte_array.astype(np.uint8))

    file_name = '%s-%02d-%d.jpg' % (basename, index, image.label)
    file = os.path.join(path, file_name)

    with open(file, mode='wb') as out:
        imageshow.save(out, format='JPEG')

raise
#

# write training data to tf_record
# FLAGS = tf.app.flags.FLAGS
# tf.app.flags.DEFINE_string('file', './data/cifar-10-batches-bin/', "path")
# tf.app.flags.DEFINE_integer('offset', 0, "start index")
# tf.app.flags.DEFINE_integer('length', 10000, "end index")
#
# if not os.path.exists('./data_tf'):
#     os.mkdir('./data_tf')
#
# record_filename = './data_tf/train.tfrecord'
# with tf.python_io.TFRecordWriter(record_filename) as tfrecord_writer:
#     basename = os.path.basename(FLAGS.file)
#     stop = FLAGS.offset + FLAGS.length
#     print ('start to convert training data')
#     # reader = Cifar10Reader(FLAGS.file)
#     for i in range(1,6):
#         reader = Cifar10Reader(FLAGS.file + 'data_batch_%d.bin'%i)
#         for index in range(FLAGS.offset, stop):
#             image = reader.read(index)
#             imageshow = Image.fromarray(image.byte_array.astype(np.uint8))
#             img_raw = np.asarray(imageshow).tostring()
#             label_raw = np.zeros([10,])
#             label_raw[int(image.label)] = 1
#             label_raw = label_raw.astype(np.uint8).tostring()
#             example = to_tfexample_raw(img_raw,label_raw)
#             tfrecord_writer.write(example.SerializeToString())
#
#         reader.close()
#     tfrecord_writer.close()
# raise


#test for read tf_record
record_iterator = tf.python_io.tf_record_iterator(path = './data_tf/train.tfrecord')
count = 0
for string_record in record_iterator:

    example = tf.train.Example()
    example.ParseFromString(string_record)

    label = (example.features.feature['label'].bytes_list.value[0])
    label_1d = np.fromstring(label, dtype = np.uint8)
    label_f = label_1d.reshape((10,))
    print (label_f)

    img = (example.features.feature['image'].bytes_list.value[0])
    img_1d = np.fromstring(img,dtype = np.uint8)
    #
    img_f = img_1d.reshape((32,32,-1))
    #
    #
    # print (img_f)
    # raise
    cv2.imshow('img-f',img_f)
    cv2.waitKey(0)
raise
