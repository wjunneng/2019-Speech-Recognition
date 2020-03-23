# -*- coding:utf-8 -*-
import os
import sys

os.chdir(sys.path[0])
from pathlib import Path
from src.confs import arguments


class DataUtils(object):
    def __init__(self):
        pass

    @staticmethod
    def get_dict_txt():
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