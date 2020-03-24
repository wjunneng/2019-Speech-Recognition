# -*- coding: utf-8 -*-
import os
import sys

os.chdir(sys.path[0])

from src.cores.cnn_ctc_am import CNNCTCAM
from src.confs import arguments


class AM(object):
    def __init__(self):
        self.cnn_model, self.ctc_model = CNNCTCAM(arguments).build()

    def train(self):
        pass
