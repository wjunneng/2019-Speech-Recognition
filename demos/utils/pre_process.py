import os
from tqdm import tqdm
import codecs

from configurations.constant import Constant
from demos.utils.util import write_pkl


class PreProcess(object):
    def __init__(self, configuration):
        configuration = configuration
        # aishell_speech/libri_speech/life_speech
        self.datasource_type = configuration['datasource_type']
        # dev/train/test文件夹保存的路径 数据集的路径
        self.path = configuration[self.datasource_type]['path']
        # label保存的路径
        self.audio_label_path = configuration[self.datasource_type]['audio_label_path']
        # audio路径与label之间的分隔符
        self.audio_label_splitter = configuration[self.datasource_type]['audio_label_splitter']

        # <pad>
        self.pad = configuration['token']['PAD']
        # <pad> flag
        self.pad_flag = configuration['token']['PAD_FLAG']
        # <unk>
        self.unk = configuration['token']['UNK']
        # <unk> flag
        self.unk_flag = configuration['token']['UNK_FLAG']
        # <sos>
        self.sos = configuration['token']['SOS']
        # <sos> flag
        self.sos_flag = configuration['token']['SOS_FLAG']
        # <eos>
        self.eos = configuration['token']['EOS']
        # <eos> flag
        self.eos_flag = configuration['token']['EOS_FLAG']
        # <space>
        self.space = configuration['token']['SPACE']
        # <space> flag
        self.space_flag = configuration['token']['SPACE_FLAG']

        # 字典
        self.vocab_to_index = {self.pad_flag: self.pad,
                               self.unk_flag: self.unk,
                               self.sos_flag: self.sos,
                               self.eos_flag: self.eos,
                               self.space_flag: self.space}
        # 反字典
        self.index_to_vocab = dict(zip(self.vocab_to_index.values(), self.vocab_to_index.keys()))

        # token_index 和 wav文件的绝对路径
        self.samples = []
        # {'train': samples, 'dev': samples, 'test': test}
        self.data = {}

    def get_data(self, type):
        """
        获取数据
        :return:
        """
        # key保存audio路径, value保存label
        audio_label = {}

        # 遍历文件
        with codecs.open(os.path.join(self.path, self.audio_label_path)) as file:
            for line in file.readlines():
                audio_label_list = line.split(self.audio_label_splitter)
                audio_label[audio_label_list[0]] = self.audio_label_splitter.join(audio_label_list[1:]).strip('\n')

        if self.datasource_type == 'aishell_speech':
            # 遍历train dev test文件夹
            floder = os.path.join(self.path, 'wav', type)
        else:
            floder = os.path.join(self.path, type)
        assert (os.path.isdir(floder) is True)

        print('len:', len(os.listdir((floder))))
        for d in tqdm(os.listdir(floder)):
            dirs = os.path.join(floder, d)
            if os.path.isdir(dirs):
                files = [file for file in os.listdir(dirs) if file.endswith('.wav')]

                for file in files:
                    file_path = os.path.join(dirs, file)
                    self.add_token(audio_label, file, file_path)

            elif os.path.isfile(dirs) and dirs.endswith('.wav'):
                self.add_token(audio_label, d, dirs)

        self.data[type] = self.samples
        self.samples = []

    def add_token(self, audio_label, file, file_path):
        """
        输入文件
        :param audio_label:
        :param file:
        :return:
        """
        token_to_index = []

        # 文件名
        if file in audio_label or file.split('.')[0] in audio_label:
            if file in audio_label:
                value = audio_label[file]
            elif file.split('.')[0] in audio_label:
                value = audio_label[file.split('.')[0]]
            # 添加<eos>标记
            value = list(value.strip()) + [self.eos_flag]

            # 遍历tokens
            for token in value:
                if token not in list(self.vocab_to_index.keys()):
                    self.vocab_to_index[token] = len(self.vocab_to_index)

                token_to_index.append(self.vocab_to_index[token])

            self.samples.append({'token_index': token_to_index, 'wav_path': file_path})


if __name__ == '__main__':
    configuration = Constant().get_configuration()

    pre_process = PreProcess(configuration=configuration)
    for type in ['train', 'test', 'dev']:
        pre_process.get_data(type)

        print('%s' % type)
        print(len(pre_process.data[type]))

    pre_process.data['VOCAB'] = pre_process.vocab_to_index
    pre_process.data['IVOCAB'] = dict(zip(pre_process.vocab_to_index.values(), pre_process.vocab_to_index.keys()))

    print('VOCAB.size: %d' % len(pre_process.data['VOCAB']))
    write_pkl(data=pre_process.data, path=os.path.join(configuration[configuration['datasource_type']]['path'],
                                                       configuration[configuration['datasource_type']][
                                                           'audio_index_pkl_path']))

    print('end!')
