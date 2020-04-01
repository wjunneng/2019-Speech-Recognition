# -*- coding: utf-8 -*-
import os
import sys

os.chdir(sys.path[0])
from src.confs import arguments
from src.libs.data_utils import SpeechData
from src.libs.model_utils import ModelUtils


class AM(object):
    def __init__(self, args):
        self.nn_type = 'cnn'
        self.args = args
        self.batch_size = self.args.am_batch_size
        self.label_sequence_length = self.args.am_label_sequence_length
        self.audio_length = self.args.am_audio_length
        self.audio_feature_length = self.args.am_audio_feature_length
        self.save_step = self.args.am_save_step
        self.epoch = self.args.am_epoch
        self.PARTICIPLE = self.args.PARTICIPLE
        self.AM_DIR = self.args.AM_DIR

        self.SpeechData = SpeechData(args=self.args)

        if self.nn_type == 'cnn':
            from src.cores.cnn_ctc_am import CNNCTCAM

            self.a_model, self.m_model = CNNCTCAM(self.args).build()
        elif self.nn_type == 'gru':
            from src.cores.gru_ctc_am import GRUCTCAM

            self.a_model, self.m_model = GRUCTCAM(self.args).build()

    def train(self):
        self.SpeechData.participle = self.PARTICIPLE

        train_dev_generator = self.SpeechData.data_generator(use_type=['train', 'dev'],
                                                             batch_size=self.batch_size,
                                                             label_sequence_length=self.label_sequence_length,
                                                             audio_length=self.audio_length,
                                                             audio_feature_length=self.audio_feature_length)

        # 迭代轮数
        for epoch in range(self.epoch):
            print('>>> epoch: {}'.format(epoch))
            self.m_model.fit_generator(generator=train_dev_generator, steps_per_epoch=self.save_step)

            self.SpeechData.use_type = 'train'

            if (epoch + 1) % 50 == 0:
                ModelUtils.model_evaluate(cnn_model=self.a_model,
                                          speech_data=self.SpeechData,
                                          audio_length=self.audio_length,
                                          audio_feature_length=self.audio_feature_length,
                                          label_sequence_length=self.label_sequence_length)

                self.SpeechData.use_type = 'dev'
                ModelUtils.model_evaluate(cnn_model=self.a_model,
                                          speech_data=self.SpeechData,
                                          audio_length=self.audio_length,
                                          audio_feature_length=self.audio_feature_length,
                                          label_sequence_length=self.label_sequence_length)

                if os.path.exists(self.AM_DIR) is False:
                    os.makedirs(self.AM_DIR)
                self.a_model.save(filepath=self.AM_DIR.joinpath(self.nn_type + '_' + str(epoch) + '.h5'))
                self.m_model.save(
                    filepath=self.AM_DIR.joinpath(self.nn_type + '_' + 'ctc_' + str(epoch) + '.h5'))

# if __name__ == '__main__':
#     AM(args=arguments).train()
