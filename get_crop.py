from __future__ import division
import math
import json
import random
import pprint
import scipy.misc
from scipy.misc import imsave
from glob import glob
import os
import numpy as np

def center_crop(x, crop_h, crop_w,
                resize_h=64, resize_w=64):
  if crop_w is None:
    crop_w = crop_h
  h, w = x.shape[:2]
  j = int(round((h - crop_h)/2.))
  i = int(round((w - crop_w)/2.))
  return scipy.misc.imresize(
      x[j:j+crop_h, i:i+crop_w], [resize_h, resize_w])


file_path = '../DCGAN-tensorflow/data/celebA'
file_list = glob(os.path.join(file_path,'*jpg'))
save_path = './train_data/celebA_crop/'

if not os.path.exists(save_path):
    os.mkdir(save_path)

for i in range(12000):
    img = scipy.misc.imread(file_list[i]).astype(np.float)
    cropped_img = center_crop(img,70,70,48,48)
    imsave('%s%05d.jpg'%(save_path,i),cropped_img)
    if i %10 == 0:
        print ('processing ...')
