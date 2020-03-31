# -*- coding:utf-8 -*-
import os
import sys

os.chdir(sys.path[0])
from tensorflow.contrib.keras import layers, models, backend, optimizers

from src.libs.data_utils import DataUtils


class MARKOVLM(object):
    def __init__(self, args):
        self.args = args

    def load_model(self):
        self.dict_pinyin = DataUtils.get_pinyin_dict()
        self.model1 = DataUtils.get_lm(self.args.LM_1_PATH)
        self.model2 = DataUtils.get_lm(self.args.LM_2_PATH)
        self.pinyin = DataUtils.get_lm_pinyin(self.args.DICT_PINYIN_PATH)

        return self.dict_pinyin, self.model1, self.model2

    def speech_to_text(self, list_syllable):
        """
        语音识别专用的处理函数

        实现从语音拼音符号到最终文本的转换

        使用恐慌模式处理一次解码失败的情况
        """
        length = len(list_syllable)
        # 传入的参数没有包含任何拼音时
        if length == 0:
            return ''

        # 存储剩余的拼音序列
        lst_syllable_remain = []
        str_result = ''

        # 存储临时输入拼音序列
        tmp_list_syllable = list_syllable

        while len(tmp_list_syllable) > 0:
            # 进行拼音转汉字解码，存储临时结果
            tmp_lst_result = self.decode(tmp_list_syllable, 0.0)

            # 有结果，不用恐慌
            if len(tmp_lst_result) > 0:
                str_result = str_result + tmp_lst_result[0][0]

            # 没结果，开始恐慌
            while len(tmp_lst_result) == 0:
                # 插入最后一个拼音
                lst_syllable_remain.insert(0, tmp_list_syllable[-1])
                # 删除最后一个拼音
                tmp_list_syllable = tmp_list_syllable[:-1]
                # 再次进行拼音转汉字解码
                tmp_lst_result = self.decode(tmp_list_syllable, 0.0)

                if len(tmp_lst_result) > 0:
                    # 将得到的结果加入进来
                    str_result = str_result + tmp_lst_result[0][0]

            # 将剩余的结果补回来
            tmp_list_syllable = lst_syllable_remain
            # 清空
            lst_syllable_remain = []

        return str_result

    def decode(self, list_syllable, yuzhi=0.0001):
        """
        实现拼音向文本的转换
        基于马尔可夫链
        """
        list_words = []

        num_pinyin = len(list_syllable)
        # 开始语音解码
        for i in range(num_pinyin):
            # 如果这个拼音在汉语拼音字典里的话
            if list_syllable[i] in self.dict_pinyin:
                # 获取拼音下属的字的列表，ls包含了该拼音对应的所有的字
                ls = self.dict_pinyin[list_syllable[i]]
            else:
                break

            if i == 0:
                # 第一个字做初始处理
                num_ls = len(ls)
                for j in range(num_ls):
                    # 设置马尔科夫模型初始状态值
                    # 设置初始概率，置为1.0
                    tuple_word = [ls[j], 1.0]
                    # 添加到可能的句子列表
                    list_words.append(tuple_word)
                continue
            else:
                # 开始处理紧跟在第一个字后面的字
                list_words_2 = []
                num_ls_word = len(list_words)
                # print('ls_wd: ',list_words)
                for j in range(0, num_ls_word):

                    num_ls = len(ls)
                    for k in range(0, num_ls):
                        # 把现有的每一条短语取出来
                        tuple_word = list(list_words[j])
                        # 尝试按照下一个音可能对应的全部的字进行组合
                        tuple_word[0] = tuple_word[0] + ls[k]

                        # 取出用于计算的最后两个字
                        tmp_words = tuple_word[0][-2:]
                        # 判断它们是不是再状态转移表里
                        if tmp_words in self.model2:
                            # print(tmp_words,tmp_words in self.model2)
                            tuple_word[1] = tuple_word[1] * float(self.model2[tmp_words]) / float(
                                self.model1[tmp_words[-2]])
                        # 核心！在当前概率上乘转移概率，公式化简后为第n-1和n个字出现的次数除以第n-1个字出现的次数
                        else:
                            tuple_word[1] = 0.0
                            continue
                        if tuple_word[1] >= pow(yuzhi, i):
                            # 大于阈值之后保留，否则丢弃
                            list_words_2.append(tuple_word)

                list_words = list_words_2
        for i in range(0, len(list_words)):
            for j in range(i + 1, len(list_words)):
                if list_words[i][1] < list_words[j][1]:
                    tmp = list_words[i]
                    list_words[i] = list_words[j]
                    list_words[j] = tmp

        return list_words
