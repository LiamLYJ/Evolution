from __future__ import print_function
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import scipy.misc

def setup_updates(flags):
    updates = {'e': {}, 'g': {}}

    updates['e']['num_updates'] = int(opt.e_updates.split(';')[0])
    updates['e'].update({x.split(':')[0]: float(x.split(':')[1])
                         for x in opt.e_updates.split(';')[1].split(',')})

    updates['g']['num_updates'] = int(opt.g_updates.split(';')[0])
    updates['g'].update({x.split(':')[0]: float(x.split(':')[1])
                         for x in opt.g_updates.split(';')[1].split(',')})
    return updates

def save(saver, session, flags, step):
    dataset_name = flags.dataset_name
    checkpoint_dir = flags.checkpoint_dir
    checkpoint_dir = os.path.join(checkpoint_dir, dataset_name)
    model_name = "Evolution.model"
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
    input_width = flags.input_width
    input_height = flags.input_height
    input_channel = flags.channel
    batch_size = flags.batch_size

    num_preprocess_threads = 1
    min_queue_examples = 256
    image_reader = tf.WholeFileReader()

    file_list = glob(os.path.join("./train_data", dataset_name, input_fname_pattern))

    filename_queue = tf.train.string_input_producer(file_list[:])
    _,image_file = image_reader.read(filename_queue)
    image = tf.image.decode_jpeg(image_file)
    image = tf.cast(tf.reshape(image,shape = [input_height, input_width, channel]), dtype = tf.float32)

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

def save_images(flags, img, iter, type):
    batch_size = flags.batch_size
    sample_dir = flags.sample_dir
    manifold_h = int(np.ceil(np.sqrt(batch_size)))
    manifold_w = int(np.floor(np.sqrt(batch_size)))
    imsave(img, [manifold_h, manifold_w], './{}/{}_{0:6d}.png'.format(sample_dir,type,iter))
