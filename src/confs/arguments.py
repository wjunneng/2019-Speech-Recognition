# -*- coding:utf-8 -*-
import os
import sys

os.chdir(sys.path[0])
from pathlib import Path

assert sys.platform.lower() == 'linux'

PWD = '/'.join(os.getenv('PWD').split('/')[:-1])

print('pwd: {}'.format(PWD))
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
LM_1_PATH = DATASETS_DIR.joinpath('lm_1.txt')
LM_2_PATH = DATASETS_DIR.joinpath('lm_2.txt')
DICT_PINYIN_PATH = DATASETS_DIR.joinpath('dict_pinyin.txt')

# 数据集
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

# 声学模型 am
am_nn_type = 'cnn'
am_vocab_size = 1426
am_batch_size = 5
am_epoch = 1000
am_save_step = 900 / 5
am_label_sequence_length = 64
am_audio_length = 1600
am_audio_feature_length = 200

# 语言模型 lm
lm_feature_dim = 200
lm_epochs = 100
lm_batch_size = 1
lm_num_heads = 8
lm_num_blocks = 6
lm_position_max_length = 100
lm_hidden_units = 512
lm_lr = 0.0003
lm_dropout_rate = 0.2
lm_is_training = True
lm_count = 5000
