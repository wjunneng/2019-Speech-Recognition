# -*- coding:utf-8 -*-
import os
import sys

os.chdir(sys.path[0])
from pathlib import Path

assert sys.platform.lower() == 'linux'

PWD = '/'.join(os.getenv('PWD').split('/')[:-2])

PROJECT_DIR = Path(PWD)
DATASETS_DIR = PROJECT_DIR.joinpath('datasets')
SRC_DIR = PROJECT_DIR.joinpath('src')
CONFS_DIR = SRC_DIR.joinpath('confs')
CORES_DIR = SRC_DIR.joinpath('cores')
LIBS_DIR = SRC_DIR.joinpath('libs')

# 字典的路径
DICT_TXT_PATH = DATASETS_DIR.joinpath('dict.txt')

# 声学模型 am
lr = 0.0008
vocab_size = 1426
batch_size = 6
label_sequence_length = 64
audio_length = 1600
audio_feature_length = 200
is_training = True
