import os
import re
import numpy as np
import librosa
from python_speech_features import mfcc
from pathlib import Path
import yaml


class Util(object):
    def __init__(self, constant_path):
        with open(constant_path) as file:
            constant = yaml.load(file.read())

        if constant['datasource_type'] is 'en':
            dir_path = constant['en_libri_speech_path']
        else:
            dir_path = constant['cn_life_speech_path']

        self.txts, self.audios, self.audio_paths = self.load_data(dir_path=dir_path)

    def audio_to_input_vector(self, audio_filename, num_cep, num_context):
        """
        音频编码
        :param audio_filename:
        :param num_cep:
        :param cum_context:
        :return:
        """
        audio, fs = librosa.load(audio_filename)

        # 获取mfcc系数
        features = mfcc(audio, samplerate=fs, numcep=num_cep, nfft=551)
        # 我们仅仅保留第二个特征 （BiRNN stride = 2）
        features = features[::2]
        # 输入中每个时间步长迈出一大步
        num_strides = len(features)
        # 初始化空值对于最终的上下文
        empty_context = np.zeros((num_context, num_cep), dtype=features.dtype)

        features = np.concatenate((empty_context, features, empty_context))
        # 创建一个具有重叠大小步幅的数组视图
        # num_context(past) + 1(present) + num_context(feature)
        window_size = 2 * num_context + 1
        train_inputs = np.lib.stride_tricks.as_strided(
            features,
            (num_strides, window_size, num_cep),
            (features.strides[0], features.strides[0], features.strides[1]),
            writeable=False
        )
        # 展开第二和第三个维度
        train_inputs = np.reshape(train_inputs, [num_strides, -1])
        # 复制数组，以便我们可以安全地对其进行写入
        train_inputs = np.copy(train_inputs)
        train_inputs = (train_inputs - np.mean(train_inputs)) / np.std(train_inputs)

        # 返回
        return train_inputs

    def load_data(self, dir_path, how_many=1):
        """
        加载数据
        :param dir_path:
        :param how_many:
        :return:
        """
        dir_path = Path(dir_path)
        txt_list = [f for f in dir_path.glob(pattern='**/*.txt') if f.is_file()]
        audio_list = [f for f in dir_path.glob(pattern='**/*.flac') if f.is_file()]

        print('Number of audio txt paths:', len(txt_list))
        print('Number of audio audio paths:', len(audio_list))

        txts = []
        audios = []
        audios_paths = []

        for txt in txt_list:
            with open(file=txt) as txt_file:
                txt_parent = txt.parent
                for line in txt_file.readlines():
                    audio_path = os.path.join(txt_parent, line.split(' ')[0] + '.flac')
                    txts.append(re.sub(r'[^A-Za-z]', ' ', ' '.join(line.split(' ')[1:])).strip())
                    audios.append(self.audio_to_input_vector(audio_path, 26, 9))
                    audios_paths.append(audio_path)

        return txts, audios, audios_paths
