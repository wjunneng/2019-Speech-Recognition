import os
import tarfile


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
