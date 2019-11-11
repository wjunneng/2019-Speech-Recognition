import os
import re
import numpy as np
import librosa
from python_speech_features import mfcc
from pathlib import Path

from sklearn.metrics import accuracy_score

from configurations.constant import Constant
import pickle


class Util(object):
    def __init__(self):
        self.constant = Constant().get_configuration()

        if self.constant['datasource_type'] == 'en':
            dir_path = self.constant['en']['path']
        else:
            dir_path = self.constant['cn']['path']

        self.txts, self.audios, self.audio_paths = self.load_data(dir_path=dir_path, how_many=self.constant['en'][
                                                                      'how_many'])
        self.txts_splitted, self.unique_chars, self.char2ind, self.ind2char, self.txts_converted = self.process_txts(
            txts=self.txts)

        self.audios_sorted, self.txts_sorted, self.audio_paths_sorted, self.txts_splitted_sorted, \
        self.txts_converted_sorted = self.sort_by_index(
            audios=self.audios, txts=self.txts, audio_paths=self.audio_paths,
            txts_splitted=self.txts_splitted, txts_converted=self.txts_converted,
            by_txt_length=False)

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

        for txt in txt_list[:how_many]:
            with open(file=txt) as txt_file:
                txt_parent = txt.parent
                for line in txt_file.readlines():
                    audio_path = os.path.join(txt_parent, line.split(' ')[0] + '.flac')
                    txts.append(re.sub(r'[^A-Za-z]', ' ', ' '.join(line.split(' ')[1:])).strip())
                    audios.append(self.audio_to_input_vector(audio_path, 26, 9))
                    audios_paths.append(audio_path)

        return txts, audios, audios_paths

    def split_txts(self, txts):
        """
        划分 txts
        :param txts:
        :return:
        """
        txts_splitted = []
        unique_chars = set()

        for txt in txts:
            splitted = list(txt)
            splitted = [char if char != ' ' else self.constant['token']['SPACE_FLAG'] for char in splitted]
            txts_splitted.append(splitted)
            unique_chars.update(splitted)

        return txts_splitted, unique_chars

    def create_lookup_dicts(self, unique_chars):
        """
        构建 字符-索引/索引-字符 字典
        :param unique_chars:
        :return:
        """
        char2ind = {}
        ind2char = {}
        index = 0

        # 已包含空格字符
        for token in self.constant['token']['flag_list']:
            char2ind[token] = index
            ind2char[index] = token
            index += 1

        # 剔除空格字符
        unique_chars.remove(self.constant['token']['SPACE_FLAG'])
        for token in unique_chars:
            char2ind[token] = index
            ind2char[index] = token
            index += 1

        return char2ind, ind2char

    def convert_char_to_index(self, txt, char2ind, bos=True, eos=True):
        """
        添加 bos/eos
        :param txt:
        :param char2ind:
        :param bos:
        :param eos:
        :return:
        """
        txt_to_inds = [char2ind[char] for char in txt]
        if bos is True:
            txt_to_inds.insert(0, self.constant['token']['BOS'])
        if eos is True:
            txt_to_inds.append(self.constant['token']['BOS'])

        return txt_to_inds

    def process_txts(self, txts):
        """
        处理 txts, 分割 txts,
        :param txts:
        :return:
        """
        txts_splitted, unique_chars = self.split_txts(txts=txts)
        char2ind, ind2char = self.create_lookup_dicts(unique_chars=unique_chars)
        # 字符序列转索引序列, 同时添加<bos>/<eos>
        txts_converted = [self.convert_char_to_index(txt=txt, char2ind=char2ind) for txt in txts_splitted]

        return txts_splitted, unique_chars, char2ind, ind2char, txts_converted

    def sort_by_index(self, audios, txts, audio_paths, txts_splitted, txts_converted, by_txt_length=True):
        """
        根据序列长度 从小到大排序
        :param audios:
        :param txts:
        :param audio_paths:
        :param txts_splitted:
        :param txts_converted:
        :param by_txt_length:
        :return:
        """
        if by_txt_length:
            indices = [txt[0] for txt in sorted(enumerate(txts_converted), key=lambda x: len(x[1]))]
        else:
            indices = [a[0] for a in sorted(enumerate(audios), key=lambda x: x[1].shape[0])]

        txts_sorted = np.array(txts)[indices]
        audios_sorted = np.array(audios)[indices]
        audio_paths_sorted = np.array(audio_paths)[indices]
        txts_splitted_sorted = np.array(txts_splitted)[indices]
        txts_converted_sorted = np.array(txts_converted)[indices]

        return audios_sorted, txts_sorted, audio_paths_sorted, txts_splitted_sorted, txts_converted_sorted

    def convert_inds_to_txt(self, inds, ind2char):
        """
        索引序列转文字序列
        :param inds:
        :param ind2char:
        :return:
        """
        inds_to_txt = [ind2char[ind] for ind in inds]
        return inds_to_txt

    def preds2txt(self, preds, ind2char, beam=False):
        """
        预测结果转文字序列, 同时, 去除 <bos> 和 <eos> 字符.
        :param preds:
        :param ind2char:
        :param beam:
        :return:
        """
        if beam:
            p2t = []
            for batch in preds:
                for sentence in batch:
                    converted_sentence = []
                    for p in sentence:
                        converted_ch = ind2char[p[0]]
                        if converted_ch != self.constant['token']['EOS_FLAG'] and converted_ch != \
                                self.constant['token']['SOS_FLAG']:
                            converted_sentence.append(converted_ch)
                    p2t.append(converted_sentence)
        else:
            p2t = []
            for batch in preds:
                for sentence in batch:
                    converted_sentence = self.convert_inds_to_txt(sentence, ind2char)
                    converted_sentence = [ch for ch in converted_sentence
                                          if ch != self.constant['token']['EOS_FLAG'] and ch != self.constant['token'][
                                              'SOS_FLAG']]
                    p2t.append(converted_sentence)

        return p2t

    def print_samples(self, preds, targets):
        """
        打印 accuracy
        :param preds:
        :param targets:
        :return:
        """
        accs = []
        for p, t in zip(preds, targets):
            if len(p) >= len(t):
                acc_score = accuracy_score(p[:len(t)], t)
            else:
                acc_score = accuracy_score(p, t[:len(p)]) - 0.3

            accs.append(acc_score)
            print('Created:', p)
            print('Actual:', t)
            print('Accuracy score:', acc_score, '\n\n')

        print('Mean acc score:', np.mean(accs))

    def write_pkl(self, path, data):
        """
        保存pkl模型
        :param path:
        :param data:
        :return:
        """
        with open(path, 'wb') as file:
            pickle.dump(data, file, pickle.HIGHEST_PROTOCOL)

    def load_pkl(self, path):
        """
        加载pkl模型
        :param path:
        :return:
        """
        with open(path, 'rb') as file:
            data = pickle.load(file)

        return data


if __name__ == "__main__":
    tmp = Util()
    print(tmp)
