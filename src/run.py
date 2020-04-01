# -*- coding:utf-8 -*-
import os
import sys

os.chdir(sys.path[0])
import numpy as np
from tensorflow.contrib.keras import backend

from src.libs.feature_utils import AudioFeature
from src.libs.model_utils import ModelUtils
from src.cores.cnn_ctc_am import CNNCTCAM
from src.libs.data_utils import DataUtils
from src.confs import arguments
from src.cores.markov_lm import MARKOVLM
from src.cores.run_am import AM

am = CNNCTCAM(args=arguments)
am.build()

AM(args=arguments).train()

# am.cnn_model.load_weights('/home/wjunneng/Ubuntu/2019-Speech-Recognition/models/am/cnn_199.h5')
# am.ctc_model.load_weights('/home/wjunneng/Ubuntu/2019-Speech-Recognition/models/am/ctc_199.h5')
#
# wav_path = '/home/wjunneng/Ubuntu/2019-Speech-Recognition/datasets/AISHELL/test/BAC009S0912W0128.wav'
# data_input = AudioFeature(audio_path=wav_path).get_original_feature()
# input_length = len(data_input)
# input_length = input_length // 8
#
# data_input = np.array(data_input, dtype=np.float)
# data_input = data_input.reshape(data_input.shape[0], data_input.shape[1], 1)
# current = ModelUtils.model_predict(cnn_model=am.cnn_model, data_feature=data_input, feature_len=input_length,
#                                    audio_length=arguments.am_audio_length,
#                                    audio_feature_length=arguments.am_audio_feature_length)
#
# print(current)
#
# list_symbol_dict = DataUtils.get_symbol_dict(path='/home/wjunneng/Ubuntu/2019-Speech-Recognition/datasets/dict.txt')
# result = []
# for i in current:
#     result.append(list(list_symbol_dict.keys())[i])
# print('result: {}'.format(result))
#
# backend.clear_session()
#
# lm = MARKOVLM(args=arguments)
# lm.load_model()
# r = lm.speech_to_text(result)
# print('语音转文字结果： {}'.format(r))
