# -*- coding:utf-8 -*-
import os
import sys

os.chdir(sys.path[0])
import jieba
import random
import numpy as np
import typing
import difflib
from pypinyin import Style
from pypinyin.contrib.neutral_tone import NeutralToneWith5Mixin
from pypinyin.converter import DefaultConverter
from pypinyin.core import Pinyin

from src.confs import arguments
from src.libs.feature_utils import AudioFeature


class DataUtils(object):
    @staticmethod
    def get_pinyin_dict():
        """
        获取字典中的数据
        :return:
        """
        result = {}

        with open(arguments.DICT_TXT_PATH, 'r', encoding='UTF-8') as file:
            for line in file.readlines():
                line = line.strip('\n').split('\t')
                result[line[0]] = line[-1]
        result['_'] = ''

        return result

    @staticmethod
    def get_edit_distance(str1, str2):
        leven_cost = 0
        s = difflib.SequenceMatcher(None, str1, str2)
        for tag, i1, i2, j1, j2 in s.get_opcodes():
            if tag == 'replace':
                leven_cost += max(i2 - i1, j2 - j1)
            elif tag == 'insert':
                leven_cost += (j2 - j1)
            elif tag == 'delete':
                leven_cost += (i2 - i1)

        return leven_cost

    @staticmethod
    def get_lm(path):
        """
        读取语言模型的文件
        返回读取后的模型
        """
        # 初始化符号字典
        dic_model = {}
        # 打开文件并读入
        with open(path, 'r', encoding='utf-8') as file:
            txt_lines = file.readlines()

            for line in txt_lines:
                if line != '':
                    txt_l = line.split('\t')
                    if len(txt_l) == 1:
                        continue
                    dic_model[txt_l[0]] = txt_l[1]

        return dic_model

    @staticmethod
    def get_lm_pinyin(path):
        """
        获取语言模型字典
        :param path:
        :return:
        """
        dic = {}

        # 打开文件并读入
        with open(path, 'r', encoding='utf-8') as file:
            txt_lines = file.readlines()

            for line in txt_lines:
                if line == '':
                    continue
                pinyin_split = line.split('\t')
                list_pinyin = pinyin_split[0]
                if list_pinyin not in dic and int(pinyin_split[1]) > 1:
                    dic[list_pinyin] = pinyin_split[1]
        return dic

    @staticmethod
    def get_symbol_dict(path):
        """
        读取拼音汉字的字典文件
        返回读取后的字典
        """
        dic = {}

        # 打开文件并读入
        with open(path, 'r', encoding='utf-8') as file:
            txt_lines = file.readlines()

            for line in txt_lines:
                # 初始化符号列表
                list_symbol = []
                pinyin = ''
                if line != '':
                    txt_l = line.split('\t')
                    pinyin = txt_l[0]
                    for word in txt_l[1]:
                        list_symbol.append(word)
                dic[pinyin] = list_symbol
        return dic


class MyConverter(NeutralToneWith5Mixin, DefaultConverter):
    pass


class SpeechData(object):
    def __init__(self, args):
        self.args = args
        self._dataset_path = self.args.DATASET_DIR

    @property
    def use_type(self):
        return self._use_type

    @use_type.setter
    def use_type(self, value: typing.Union[list, str]):
        if isinstance(value, str):
            if value not in ['train', 'dev', 'test']:
                raise ValueError('use_type: {} not in ["train", "test", "dev"]!'.format(value))
            self._use_type = [value]

        elif isinstance(value, list):
            for item in value:
                if item not in ['train', 'dev', 'test']:
                    raise ValueError('use_type: {} not in ["train", "test", "dev"]!'.format(value))
            self._use_type = value

        else:
            raise TypeError('use_type: {} not in ["str", "list"]!'.format(value))

    @property
    def participle(self):
        return self.participle

    @participle.setter
    def participle(self, value: bool):
        if isinstance(value, bool) is False:
            raise TypeError('participle: {} type is error!'.format(value))

        self._participle = value

    def get_audio_dict(self) -> (dict, dict, dict):
        """
        获取原始数据
        :return:
        """
        use_type = self._use_type
        dataset_path = self._dataset_path
        participle = self._participle

        id_path_dict = {}
        id_hanzi_dict = {}
        id_pinyin_dict = {}

        for use_type in use_type:
            with open(file=dataset_path.joinpath(use_type + '.txt'), mode='r', encoding='utf-8') as txt_file:
                for line in txt_file.readlines():
                    # 生成id(str)
                    id = line.split('\t')[0]

                    # 生成audio路径
                    path = dataset_path.joinpath(use_type, id)

                    # 是否需要进行分词 生成汉字(str)
                    hanzi = line.split('\t')[1].strip('\n')
                    if participle:
                        hanzi = list(jieba.cut(hanzi, cut_all=False))
                    else:
                        hanzi = hanzi.split(' ')

                    # 生成拼音(str)
                    pinyin_dict = DataUtils.get_pinyin_dict()
                    my_pinyin = Pinyin(MyConverter())
                    pinyin = ''
                    for token in hanzi:
                        for char in my_pinyin.pinyin(token, style=Style.TONE3, heteronym=False):
                            if char[0] not in pinyin_dict:
                                pinyin += ('_' + ' ')
                            else:
                                pinyin += (char[0] + ' ')

                    id_path_dict[id] = path
                    id_hanzi_dict[id] = ' '.join(list(''.join(hanzi)))
                    id_pinyin_dict[id] = pinyin

        return id_path_dict, id_hanzi_dict, id_pinyin_dict

    def data_generator(self, use_type: typing.Union[list, str], batch_size: int, label_sequence_length: int,
                       audio_length: int, audio_feature_length: int):
        """
        数据生成器
        :param use_type:
        :param batch_size:
        :param label_sequence_length:
        :param audio_length:
        :param audio_feature_length:
        :return:
        """
        self.use_type = use_type
        id_path_dict, id_hanzi_dict, id_pinyin_dict = self.get_audio_dict()
        pinyin_dict = DataUtils.get_pinyin_dict()

        labels = np.zeros((batch_size, 1), dtype=np.float)
        audio_feature = AudioFeature()

        while True:
            X = np.zeros((batch_size, audio_length, audio_feature_length, 1), dtype=np.float)
            y = np.zeros((batch_size, label_sequence_length), dtype=np.int16)

            input_length = []
            label_length = []

            for i in range(batch_size):
                # 获取一个随机key, 并获取文件名
                key = random.choice(list(id_path_dict.keys()))
                audio_feature.audio_path = id_path_dict[key]
                # 获取特征值、拼音、汉字
                data_feature = audio_feature.get_original_feature()
                data_feature = data_feature.reshape(data_feature.shape[0], data_feature.shape[1], 1)
                data_pinyin = np.asarray(
                    [list(pinyin_dict.keys()).index(i) for i in id_pinyin_dict[key].strip().split(' ')])
                data_feature = data_feature[:min(data_feature.shape[0], audio_length),
                               :min(data_feature.shape[1], audio_feature_length), :]
                # 关于下面这一行取整除以8 并加8的余数，在实际中如果遇到报错，可尝试只在有余数时+1，没有余数时+0，或者干脆都不加，只留整除
                # input_length.append(data_feature.shape[0] // 8 + data_feature.shape[0] % 8)
                input_length.append(data_feature.shape[0] // 8)

                data_pinyin = data_pinyin[:min(data_pinyin.shape[0], label_sequence_length)]
                X[i, 0:len(data_feature)] = data_feature
                y[i, 0:len(data_pinyin)] = data_pinyin
                label_length.append([len(data_pinyin)])

            label_length = np.array(label_length)
            input_length = np.array([input_length]).T

            yield [X, y, input_length, label_length], labels

# if __name__ == '__main__':
#     data = 'AISHELL'
#
#     if data == 'THCHS-30':
#         dataset_path = '/home/wjunneng/Ubuntu/2019-Speech-Recognition/datasets/THCHS-30'
#         use_type = 'train'
#         use_txt_path = '/home/wjunneng/Ubuntu/2019-Speech-Recognition/datasets/THCHS-30/train.txt'
#         participle = False
#         id_path_dict, id_hanzi_dict, id_pinyin_dict = DataUtils().get_audio_dict(dataset_path=dataset_path,
#                                                                                  use_type=use_type,
#                                                                                  use_txt_path=use_txt_path,
#                                                                                  participle=participle)
#     elif data == 'ST-CMDS':
#         dataset_path = '/home/wjunneng/Ubuntu/2019-Speech-Recognition/datasets/ST-CMDS'
#         use_type = 'train'
#         use_txt_path = '/home/wjunneng/Ubuntu/2019-Speech-Recognition/datasets/ST-CMDS/train.txt'
#         participle = True
#         id_path_dict, id_hanzi_dict, id_pinyin_dict = DataUtils().get_audio_dict(dataset_path=dataset_path,
#                                                                                  use_type=use_type,
#                                                                                  use_txt_path=use_txt_path,
#                                                                                  participle=participle)
#     elif data == 'AISHELL':
#         dataset_path = '/home/wjunneng/Ubuntu/2019-Speech-Recognition/datasets/AISHELL'
#         use_type = 'train'
#         use_txt_path = '/home/wjunneng/Ubuntu/2019-Speech-Recognition/datasets/AISHELL/train.txt'
#         participle = False
#         id_path_dict, id_hanzi_dict, id_pinyin_dict = DataUtils().get_audio_dict(dataset_path=dataset_path,
#                                                                                  use_type=use_type,
#                                                                                  use_txt_path=use_txt_path,
#                                                                                  participle=participle)
