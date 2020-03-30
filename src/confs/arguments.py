# -*- coding:utf-8 -*-
import os
import sys

os.chdir(sys.path[0])
from pathlib import Path

assert sys.platform.lower() == 'linux'

PWD = '/'.join(os.getenv('PWD').split('/')[:-2])

PROJECT_DIR = Path(PWD)
DATASETS_DIR = PROJECT_DIR.joinpath('datasets')
MODELS_DIR = PROJECT_DIR.joinpath('models')
AM_DIR = MODELS_DIR.joinpath('am')
SRC_DIR = PROJECT_DIR.joinpath('src')
CONFS_DIR = SRC_DIR.joinpath('confs')
CORES_DIR = SRC_DIR.joinpath('cores')
LIBS_DIR = SRC_DIR.joinpath('libs')

# 字典的路径
DICT_TXT_PATH = DATASETS_DIR.joinpath('dict.txt')

# 声学模型 am
# lr = 0.0008
vocab_size = 1426
batch_size = 5
epoch = 10
save_step = 900 / 5
label_sequence_length = 64
audio_length = 1600
audio_feature_length = 200

DATA_TYPE = 'AISHELL'

if DATA_TYPE == 'AISHELL':
    DATASET_DIR = DATASETS_DIR.joinpath('AISHELL')
    PARTICIPLE = False

elif DATA_TYPE == 'ST-CMDS':
    DATASET_DIR = DATASETS_DIR.joinpath('ST-CMDS')
    PARTICIPLE = True

elif DATA_TYPE == 'THCHS-30':
    DATASET_DIR = DATASETS_DIR.joinpath('THCHS-30')
    PARTICIPLE = False
