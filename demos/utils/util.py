import os
import tarfile
import pickle


def extract(filename):
    """
    解压 压缩文件
    :param filename:
    :return:
    """
    print('Extracting {}...'.format(filename))
    tar = tarfile.open(filename, 'r')
    tar.extractall('data')
    tar.close()
    print('End !!!')


def write_pkl(path, data):
    """
    保存pkl模型
    :param path:
    :param data:
    :return:
    """
    with open(path, 'wb') as file:
        pickle.dump(data, file, pickle.HIGHEST_PROTOCOL)


def load_pkl(path):
    """
    加载pkl模型
    :param path:
    :return:
    """
    with open(path, 'rb') as file:
        data = pickle.load(file)

    return data


class AverageMeter(object):
    """
    Keeps track of most recent, average, sum, and count of a metric.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
