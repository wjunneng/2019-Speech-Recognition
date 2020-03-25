# -*- coding: utf-8 -*-
import os
import sys

os.chdir(sys.path[0])
import wave
import numpy as np
from pathlib import Path
from scipy.fftpack import fft


class AudioFeature(object):
    def __init__(self, audio_path: str = ''):
        self._audio_path = audio_path

    @property
    def audio_path(self):
        return self.audio_path

    @audio_path.setter
    def audio_path(self, value):
        value = Path(value)
        if value.exists() is False:
            raise FileNotFoundError('{} file not found!'.format(value))
        if value.suffix != '.wav':
            raise ValueError('{} file does not end with .wav!'.format(value))
        self._audio_path = value

    def get_original_feature(self):
        """
        原始的方式获取特征
        :return:
        """
        # 打开一个.wav文件格式的声音文件流
        with wave.open(f=str(self._audio_path), mode='rb') as wav_file:
            # 获取帧数
            num_frame = wav_file.getnframes()
            # 获取声道数
            num_channel = wav_file.getnchannels()
            # 获取帧速率
            framerate = wav_file.getframerate()
            # 读取全部的帧， 并声音文件数据转换为数组矩阵形式
            wav_feature = np.fromstring(wav_file.readframes(nframes=num_frame), dtype=np.uint8)

        # 按照声道数将数组整形，单声道时候是一列数组，双声道时候是两列的矩阵
        wav_feature.shape = -1, num_channel
        # 将矩阵转置
        wav_feature = wav_feature.T

        if framerate != 16000:
            raise ValueError(
                '[Error] ASRT currently only supports wav audio files with a sampling rate of 16000 Hz, '
                'but this audio is ' + str(framerate) + ' Hz. ')
        x = np.linspace(0, 400 - 1, 400, dtype=np.int64)
        # 汉明窗
        w = 0.54 - 0.46 * np.cos(2 * np.pi * x / (400 - 1))

        # wav波形 加时间窗以及时移10ms 单位ms
        time_window = 25
        # 计算窗长度的公式，目前全部为400固定值
        # window_length = framerate / 1000 * time_window
        wav_length = wav_feature.shape[1]

        # 计算循环终止的位置，也就是最终生成的窗数
        range0_end = int(len(wav_feature[0]) / framerate * 1000 - time_window) // 10
        # 用于存放最终的频率特征数据
        data_input = np.zeros((range0_end, 200), dtype=np.float)
        for i in range(0, range0_end):
            p_start = i * 160
            p_end = p_start + 400
            data_line = wav_feature[0, p_start:p_end]
            # 加窗
            data_line = data_line * w
            data_line = np.abs(fft(data_line)) / wav_length
            # 设置为400除以2的值（即200）是取一半数据，因为是对称的
            data_input[i] = data_line[0:200]

        data_input = np.log(data_input + 1)

        return data_input
