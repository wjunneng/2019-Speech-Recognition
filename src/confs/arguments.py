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
# lr = 0.0008
vocab_size = 1426
batch_size = 4
epoch = 50
save_step = 100
label_sequence_length = 64
audio_length = 1600
audio_feature_length = 200

USE_TYPE = 'train'
DATA_TYPE = 'AISHELL'

if DATA_TYPE == 'AISHELL':
    DATASET_DIR = DATASETS_DIR.joinpath('AISHELL')
    PARTICIPLE = False

    if USE_TYPE == 'train':
        USE_TXT_PATH = DATASET_DIR.joinpath('train.txt')
    elif USE_TYPE == 'test':
        USE_TXT_PATH = DATASET_DIR.joinpath('test.txt')
    elif USE_TYPE == 'dev':
        USE_TXT_PATH = DATASET_DIR.joinpath('dev.txt')

elif DATA_TYPE == 'ST-CMDS':
    DATASET_DIR = DATASETS_DIR.joinpath('ST-CMDS')
    PARTICIPLE = True

    if USE_TYPE == 'train':
        USE_TXT_PATH = DATASET_DIR.joinpath('train.txt')
    elif USE_TYPE == 'test':
        USE_TXT_PATH = DATASET_DIR.joinpath('test.txt')
    elif USE_TYPE == 'dev':
        USE_TXT_PATH = DATASET_DIR.joinpath('dev.txt')

elif DATA_TYPE == 'THCHS-30':
    DATASET_DIR = DATASETS_DIR.joinpath('THCHS-30')
    PARTICIPLE = False

    if USE_TYPE == 'train':
        USE_TXT_PATH = DATASET_DIR.joinpath('train.txt')
    elif USE_TYPE == 'test':
        USE_TXT_PATH = DATASET_DIR.joinpath('test.txt')
    elif USE_TYPE == 'dev':
        USE_TXT_PATH = DATASET_DIR.joinpath('dev.txt')
