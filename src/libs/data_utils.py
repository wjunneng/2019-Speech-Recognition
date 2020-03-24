# -*- coding:utf-8 -*-
import os
import sys

os.chdir(sys.path[0])
import jieba
from pathlib import Path
from src.confs import arguments

from pypinyin import lazy_pinyin, pinyin, Style
from pypinyin.contrib.neutral_tone import NeutralToneWith5Mixin
from pypinyin.converter import DefaultConverter
from pypinyin.core import Pinyin


class MyConverter(NeutralToneWith5Mixin, DefaultConverter):
    pass


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
    def get_audio_dict(dataset_path: str, use_type: str, use_txt_path: str, participle: bool) -> (dict, dict, dict):
        """
        获取原始数据
        :param dataset_path: eg: ../AISHELL | ../ST-CMDS | ../THCHS-30
        :param use_type: eg: train | dev | test
        :param use_txt_path: eg: ../AISHELL/train.txt | ../AISHELL/test.txt
        :param participle: eg: True|False
        :return:
        """
        id_path_dict = {}
        id_hanzi_dict = {}
        id_pinyin_dict = {}
        dataset_path = Path(dataset_path)
        use_txt_path = Path(use_txt_path)
        with open(file=use_txt_path, mode='r', encoding='utf-8') as txt_file:
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


class SpeechData(object):
    def __init__(self, dataset_path: str, use_type: str, use_txt_path: str, participle: bool):
        self.dataset_path = dataset_path
        self.use_type = use_type
        self.use_txt_path = use_txt_path
        self.participle = participle

    @property
    def use_type(self):
        return self.use_type

    @use_type.setter
    def use_type(self, use_type: str):
        if use_type not in ['train', 'dev', 'test']:
            raise ValueError('use_type not in ["train", "test", "dev"]')

        self.use_type = use_type

    @property
    def participle(self):
        return self.participle

    @participle.setter
    def participle(self, participle: bool):
        if isinstance(participle, bool) is False:
            raise ValueError('participle type is not bool')
        self.participle = participle

    def get_audio_dict(self):
        return DataUtils().get_audio_dict(dataset_path=self.dataset_path, use_type=self.use_type,
                                          use_txt_path=self.use_txt_path, participle=self.participle)


if __name__ == '__main__':
    data = 'AISHELL'

    if data == 'THCHS-30':
        dataset_path = '/home/wjunneng/Ubuntu/2019-Speech-Recognition/datasets/THCHS-30'
        use_type = 'train'
        use_txt_path = '/home/wjunneng/Ubuntu/2019-Speech-Recognition/datasets/THCHS-30/train.txt'
        participle = False
        id_path_dict, id_hanzi_dict, id_pinyin_dict = DataUtils().get_audio_dict(dataset_path=dataset_path,
                                                                                 use_type=use_type,
                                                                                 use_txt_path=use_txt_path,
                                                                                 participle=participle)
    elif data == 'ST-CMDS':
        dataset_path = '/home/wjunneng/Ubuntu/2019-Speech-Recognition/datasets/ST-CMDS'
        use_type = 'train'
        use_txt_path = '/home/wjunneng/Ubuntu/2019-Speech-Recognition/datasets/ST-CMDS/train.txt'
        participle = True
        id_path_dict, id_hanzi_dict, id_pinyin_dict = DataUtils().get_audio_dict(dataset_path=dataset_path,
                                                                                 use_type=use_type,
                                                                                 use_txt_path=use_txt_path,
                                                                                 participle=participle)
    elif data == 'AISHELL':
        dataset_path = '/home/wjunneng/Ubuntu/2019-Speech-Recognition/datasets/AISHELL'
        use_type = 'train'
        use_txt_path = '/home/wjunneng/Ubuntu/2019-Speech-Recognition/datasets/AISHELL/train.txt'
        participle = False
        id_path_dict, id_hanzi_dict, id_pinyin_dict = DataUtils().get_audio_dict(dataset_path=dataset_path,
                                                                                 use_type=use_type,
                                                                                 use_txt_path=use_txt_path,
                                                                                 participle=participle)

        print('yes')
