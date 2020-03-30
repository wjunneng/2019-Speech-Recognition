# -*- coding:utf-8 -*-
import os
import sys

os.chdir(sys.path[0])
import numpy as np
from tensorflow.contrib.keras import backend
from src.libs.data_utils import DataUtils
from src.libs.data_utils import SpeechData
from src.libs.feature_utils import AudioFeature
from tensorflow.contrib.keras import models


class ModelUtils(object):

    @staticmethod
    def model_evaluate(cnn_model: models.Model, speech_data: SpeechData, audio_length: int, audio_feature_length: int,
                       label_sequence_length: int):
        id_path_dict, id_hanzi_dict, id_pinyin_dict = speech_data.get_audio_dict()
        audio_feature = AudioFeature()
        pinyin_dict = DataUtils.get_pinyin_dict()

        try:
            words_num = 0
            word_error_num = 0

            for key in id_path_dict.keys():
                audio_feature.audio_path = id_path_dict[key]
                # 获取特征值、拼音、汉字
                data_feature = audio_feature.get_original_feature()
                data_feature = data_feature.reshape(data_feature.shape[0], data_feature.shape[1], 1)
                data_pinyin = np.asarray(
                    [list(pinyin_dict.keys()).index(i) for i in id_pinyin_dict[key].strip().split(' ')])

                data_feature = data_feature[:min(data_feature.shape[0], audio_length),
                               :min(data_feature.shape[1], audio_feature_length), :]

                data_pinyin = data_pinyin[:min(data_pinyin.shape[0], label_sequence_length)]

                predict_pinyin = ModelUtils.model_predict(cnn_model=cnn_model, data_feature=data_feature,
                                                          feature_len=data_feature.shape[0] // 8,
                                                          audio_length=audio_length,
                                                          audio_feature_length=audio_feature_length)

                # 获取每个句子的字数
                words_n = data_pinyin.shape[0]
                # 把句子的总字数加上
                words_num += words_n
                print(data_pinyin)
                print(predict_pinyin)
                # 获取编辑距离
                edit_distance = DataUtils.get_edit_distance(data_pinyin, predict_pinyin)
                # 当编辑距离小于等于句子字数时
                if edit_distance <= words_n:
                    # 使用编辑距离作为错误字数
                    word_error_num += edit_distance
                else:
                    # 否则肯定是增加了一堆乱七八糟的奇奇怪怪的字, 就直接加句子本来的总字数就好了
                    word_error_num += words_n

            print('*[测试结果] 语音识别 ' + '集语音单字错误率：', word_error_num / words_num * 100, '%')
            print('*[Test Result] Speech Recognition ' + ' set word error ratio: ', word_error_num / words_num * 100,
                  '%')

        except StopIteration:
            print('[Error] Model Test Error. please check data format.')

    @staticmethod
    def model_predict(cnn_model, data_feature, feature_len, audio_length: int, audio_feature_length: int):
        """
        预测结果
        返回语音识别后的拼音符号列表
        """
        batch_size = 1
        in_len = np.zeros(shape=batch_size, dtype=np.int32)
        in_len[0] = feature_len
        x_in = np.zeros((batch_size, audio_length, audio_feature_length, 1), dtype=np.float)

        for i in range(batch_size):
            x_in[i, 0:len(data_feature)] = data_feature

        base_pred = cnn_model.predict(x=x_in)
        base_pred = base_pred[:, :, :]
        r = backend.ctc_decode(base_pred, in_len, greedy=True, beam_width=100, top_paths=1)
        r1 = backend.get_value(r[0][0])
        r1 = r1[0]

        return r1
