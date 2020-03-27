# -*- coding: utf-8 -*-
import os
import sys

os.chdir(sys.path[0])

from . import cores, confs, libs

import tensorflow as tf
from tensorflow.contrib import keras
import warnings

warnings.filterwarnings('ignore')
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
config = tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.95
config.gpu_options.allow_growth = True
config.log_device_placement = False
keras.backend.set_session(tf.compat.v1.Session(config=config))
