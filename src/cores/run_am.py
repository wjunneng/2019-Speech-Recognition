# -*- coding: utf-8 -*-
import os
import sys

os.chdir(sys.path[0])

from src.cores.cnn_ctc_am import CNNCTCAM
from src.confs import arguments
from src.libs.data_utils import SpeechData


class AM(object):
    def __init__(self, args):
        self.args = args
        self.cnn_model, self.ctc_model = CNNCTCAM(self.args).build()
        self.SpeechData = SpeechData(dataset_path=self.args.DATASET_DIR)

    def train(self):
        self.SpeechData.use_type = self.args.USE_TYPE
        self.SpeechData.use_txt_path = self.args.USE_TXT_PATH
        self.SpeechData.participle = self.args.PARTICIPLE

        self.SpeechData.set_speech_data()
        data_generator = self.SpeechData.data_generator(batch_size=self.args.batch_size,
                                                        label_sequence_length=self.args.label_sequence_length,
                                                        audio_length=self.args.audio_length,
                                                        audio_feature_length=self.args.audio_feature_length)

        # 迭代轮数
        for epoch in range(self.args.epoch):
            # 迭代数据数
            n_step = 0
            while True:
                try:
                    print('[message] epoch %d . Have train datas %d+' % (epoch, n_step * self.args.save_step))
                    self.ctc_model.fit_generator(data_generator, self.args.save_step)
                    n_step += 1
                except StopIteration:
                    print('[error] generator error. please check data format.')
                    break


if __name__ == '__main__':
    AM(args=arguments).train()
