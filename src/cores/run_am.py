# -*- coding: utf-8 -*-
import os
import sys

os.chdir(sys.path[0])

from src.cores.cnn_ctc_am import CNNCTCAM
from src.confs import arguments
from src.libs.data_utils import SpeechData
from src.libs.model_utils import ModelUtils


class AM(object):
    def __init__(self, args):
        self.args = args
        self.cnn_model, self.ctc_model = CNNCTCAM(self.args).build()
        self.SpeechData = SpeechData(args=self.args)

    def train(self):
        self.SpeechData.participle = self.args.PARTICIPLE

        train_dev_generator = self.SpeechData.data_generator(use_type=['dev', 'train'],
                                                             batch_size=self.args.batch_size,
                                                             label_sequence_length=self.args.label_sequence_length,
                                                             audio_length=self.args.audio_length,
                                                             audio_feature_length=self.args.audio_feature_length)

        # 迭代轮数
        for epoch in range(self.args.epoch):
            print('>>> epoch: {}'.format(epoch))
            self.ctc_model.fit_generator(generator=train_dev_generator, steps_per_epoch=self.args.save_step)

            self.SpeechData.use_type = 'train'

            if (epoch + 1) % 5 == 0:
                ModelUtils.model_evaluate(cnn_model=self.cnn_model,
                                          speech_data=self.SpeechData,
                                          audio_length=self.args.audio_length,
                                          audio_feature_length=self.args.audio_feature_length,
                                          label_sequence_length=self.args.label_sequence_length)

                self.SpeechData.use_type = 'dev'
                ModelUtils.model_evaluate(cnn_model=self.cnn_model,
                                          speech_data=self.SpeechData,
                                          audio_length=self.args.audio_length,
                                          audio_feature_length=self.args.audio_feature_length,
                                          label_sequence_length=self.args.label_sequence_length)

                if os.path.exists(self.args.AM_DIR) is False:
                    os.makedirs(self.args.AM_DIR)
                self.cnn_model.save(filepath=self.args.AM_DIR.joinpath('cnn_' + str(epoch) + '.h5'))
                self.ctc_model.save(filepath=self.args.AM_DIR.joinpath('ctc_' + str(epoch) + '.h5'))


if __name__ == '__main__':
    AM(args=arguments).train()
