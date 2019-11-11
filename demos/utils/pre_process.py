import os
import pickle
from tqdm import tqdm
import codecs

from configurations.constant import Constant


# from config import tran_file, pickle_file
# from utils import ensure_folder

class PreProcess(object):
    def __init__(self):
        configuration = Constant().get_configuration()

        if configuration['datasource_type'] is 'cn':
            # cn or en
            self.datasource_type = configuration['datasource_type']
            # dev/train/test文件夹保存的路径 数据集的路径
            self.path = configuration[self.datasource_type]['path']
            # label保存的路径
            self.audio_label_path = configuration[self.datasource_type]['audio_label_path']
            # audio路径与label之间的分隔符
            self.audio_label_splitter = configuration[self.datasource_type]['audio_label_splitter']
            self.get_data()
        else:
            return None

    def get_data(self):
        # key保存audio路径, value保存label
        audio_label = {}
        # 遍历文件
        with codecs.open(self.audio_label_path) as file:
            audio_label_list = file.read().split(self.audio_label_splitter)
            audio_label[audio_label_list[0]] = self.audio_label_splitter.join(audio_label_list[1:])

        # 遍历train dev test文件夹
        for type in ['train', 'dev', 'test']:
            floder = os.path.join(self.path, self.audio_label_path)
            assert os.path.isdir(floder) is True

            for d in os.listdir(floder):
                if os.path.isdir(os.path.join(floder, d)):

        def

def get_data(split):
    print('getting {} data...'.format(split))

    global VOCAB

    with open(tran_file, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    tran_dict = dict()
    for line in lines:
        tokens = line.split()
        key = tokens[0]
        trn = ''.join(tokens[1:])
        tran_dict[key] = trn

    samples = []

    folder = os.path.join(wav_folder, split)
    ensure_folder(folder)
    dirs = [os.path.join(folder, d) for d in os.listdir(folder) if os.path.isdir(os.path.join(folder, d))]
    for dir in tqdm(dirs):
        files = [f for f in os.listdir(dir) if f.endswith('.wav')]

        for f in files:
            wave = os.path.join(dir, f)

            key = f.split('.')[0]
            if key in tran_dict:
                trn = tran_dict[key]
                trn = list(trn.strip()) + ['<eos>']

                for token in trn:
                    build_vocab(token)

                trn = [VOCAB[token] for token in trn]

                samples.append({'trn': trn, 'wave': wave})

    print('split: {}, num_files: {}'.format(split, len(samples)))
    return samples


def build_vocab(token):
    global VOCAB, IVOCAB
    if not token in VOCAB:
        next_index = len(VOCAB)
        VOCAB[token] = next_index
        IVOCAB[next_index] = token


if __name__ == "__main__":
    VOCAB = {'<sos>': 0, '<eos>': 1}
    IVOCAB = {0: '<sos>', 1: '<eos>'}

    data = dict()
    data['VOCAB'] = VOCAB
    data['IVOCAB'] = IVOCAB
    data['train'] = get_data('train')
    data['dev'] = get_data('dev')
    data['test'] = get_data('test')

    with open(pickle_file, 'wb') as file:
        pickle.dump(data, file)

    print('num_train: ' + str(len(data['train'])))
    print('num_dev: ' + str(len(data['dev'])))
    print('num_test: ' + str(len(data['test'])))
    print('vocab_size: ' + str(len(data['VOCAB'])))
