from __future__ import print_function
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import scipy.misc
from glob import glob
import os

def setup_updates(flags):
    updates = {'e': {}, 'g': {}, 'd1': {}, 'd2': {}}

    updates['e']['num_updates'] = int(flags.e_updates.split(';')[0])
    updates['e'].update({x.split(':')[0]: float(x.split(':')[1])
                         for x in flags.e_updates.split(';')[1].split(',')})

    updates['g']['num_updates'] = int(flags.g_updates.split(';')[0])
    updates['g'].update({x.split(':')[0]: float(x.split(':')[1])
                         for x in flags.g_updates.split(';')[1].split(',')})

    updates['d1']['num_updates'] = int(flags.d1_updates.split(';')[0])
    updates['d1'].update({x.split(':')[0]: float(x.split(':')[1])
                         for x in flags.d1_updates.split(';')[1].split(',')})

    updates['d2']['num_updates'] = int(flags.d2_updates.split(';')[0])
    updates['d2'].update({x.split(':')[0]: float(x.split(':')[1])
                         for x in flags.d2_updates.split(';')[1].split(',')})
    return updates

def save(saver, session, flags, step):
    dataset_name = flags.dataset_name
    checkpoint_dir = flags.checkpoint_dir
    checkpoint_dir = os.path.join(checkpoint_dir, dataset_name)
    model_name = flags.model_name
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    saver.save(session,
                    os.path.join(checkpoint_dir, model_name),
                    global_step=step)

def load(saver, session, flags):
    import re
    print(" [*] Reading checkpoint...")
    dataset_name = flags.dataset_name
    checkpoint_dir = flags.checkpoint_dir
    checkpoint_dir = os.path.join(checkpoint_dir, dataset_name)

    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        saver.restore(session, os.path.join(checkpoint_dir, ckpt_name))
        counter = int(next(re.finditer("(\d+)(?!.*\d)",ckpt_name)).group(0))
        print(" [*] Success to read {}".format(ckpt_name))
        return True, counter
    else:
        print(" [*] Failed to find a checkpoint")
        return False, 0

def get_training_data(flags):
    input_fname_pattern = flags.input_fname_pattern
    input_size = flags.input_size
    channel = flags.nc
    batch_size = flags.batch_size
    dataset_name = flags.dataset_name


    num_preprocess_threads = 1
    min_queue_examples = 256
    image_reader = tf.WholeFileReader()

    file_list = glob(os.path.join("./train_data", dataset_name, input_fname_pattern))

    filename_queue = tf.train.string_input_producer(file_list[:])
    _,image_file = image_reader.read(filename_queue)
    image = tf.image.resize_images(tf.image.decode_jpeg(image_file),[input_size, input_size])
    image = tf.cast(tf.reshape(image,shape = [input_size, input_size, channel]), dtype = tf.float32)

    batch_images = tf.train.shuffle_batch([image],
                                        batch_size = batch_size,
                                        num_threads = num_preprocess_threads,
                                        capacity = min_queue_examples + 3*batch_size,
                                        min_after_dequeue = min_queue_examples)
    batch_images = batch_images /255.0
    return batch_images


def merge(images, size):
  h, w = images.shape[1], images.shape[2]
  if (images.shape[3] in (3,4)):
    c = images.shape[3]
    img = np.zeros((h * size[0], w * size[1], c))
    for idx, image in enumerate(images):
      i = idx % size[1]
      j = idx // size[1]
      img[j * h:j * h + h, i * w:i * w + w, :] = image
    return img
  elif images.shape[3]==1:
    img = np.zeros((h * size[0], w * size[1]))
    for idx, image in enumerate(images):
      i = idx % size[1]
      j = idx // size[1]
      img[j * h:j * h + h, i * w:i * w + w] = image[:,:,0]
    return img
  else:
    raise ValueError('in merge(images,size) images parameter '
                     'must have dimensions: HxW or HxWx3 or HxWx4')

def imsave(images, size, path):
    image = np.squeeze(merge(images, size))
    return scipy.misc.imsave(path,image)

def save_images(flags, img, iter, type_name):
    batch_size = flags.batch_size
    sample_dir = flags.sample_dir
    dataset_name = flags.dataset_name
    manifold_h = int(np.ceil(np.sqrt(batch_size)))
    manifold_w = int(np.floor(np.sqrt(batch_size)))
    imsave(img, [manifold_h, manifold_w], './%s/%s/%s_%06d.png'%(sample_dir, dataset_name, type_name, iter))
