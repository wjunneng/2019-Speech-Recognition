# -*- coding:utf-8 -*-
import os
import sys

os.chdir(sys.path[0])
from tensorflow.contrib.keras import layers, models, backend, optimizers


class GRUCTCAM(object):
    def __init__(self, args):
        self.OUTPUT_SIZE = args.vocab_size
        self.AUDIO_LENGTH = args.audio_length
        self.AUDIO_FEATURE_LENGTH = args.audio_feature_length
        self.LABEL_SEQUENCE_LENGTH = args.label_sequence_length

    def _ctc_lambda_func(self, args):
        y_pred, labels, input_length, label_length = args

        y_pred = y_pred[:, :, :]

        return backend.ctc_batch_cost(y_true=labels, y_pred=y_pred, input_length=input_length,
                                      label_length=label_length)

    @staticmethod
    def _bi_gru(units, x):
        x = layers.Dropout(0.2)(x)
        y1 = layers.GRU(units, return_sequences=True, kernel_initializer='he_normal',
                        recurrent_initializer='orthogonal')(x)
        y2 = layers.GRU(units, return_sequences=True, go_backwards=True, kernel_initializer='he_normal',
                        recurrent_initializer='orthogonal')(x)
        y = layers.add([y1, y2])
        return y

    @staticmethod
    def _dense(units, x, activation="relu"):
        x = layers.Dropout(0.2)(x)
        y = layers.Dense(units, activation=activation, use_bias=True,
                         kernel_initializer='he_normal')(x)
        return y

    def _gru_ctc_init(self):
        self.input_data = layers.Input(name='the_input', shape=(self.AUDIO_LENGTH, self.AUDIO_FEATURE_LENGTH, 1))
        layers_h1 = layers.Reshape((-1, 200))(self.input_data)
        layers_h2 = GRUCTCAM._dense(128, layers_h1)
        layers_h3 = GRUCTCAM._bi_gru(64, layers_h2)
        y_pred = GRUCTCAM._dense(self.OUTPUT_SIZE, layers_h3, activation='softmax')

        self.gru_model = models.Model(inputs=self.input_data, outputs=y_pred)

        self.labels = layers.Input(name='the_label', shape=[self.LABEL_SEQUENCE_LENGTH], dtype='float32')
        self.input_length = layers.Input(name='input_length', shape=[1], dtype='int64')
        self.label_length = layers.Input(name='label_length', shape=[1], dtype='int64')
        self.loss = layers.Lambda(function=self._ctc_lambda_func, output_shape=(1,), name='ctc')([y_pred,
                                                                                                  self.labels,
                                                                                                  self.input_length,
                                                                                                  self.label_length])

        self.ctc_model = models.Model(inputs=[self.input_data, self.labels, self.input_length, self.label_length],
                                      outputs=self.loss)
        optimizer = optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, decay=0.0, epsilon=10e-8)

        self.ctc_model.compile(optimizer=optimizer, loss={'ctc': lambda y_true, y_pred: y_pred})
        print('[*Info] Create Model Successful, Compiles Model Successful. ')

        return self.gru_model, self.ctc_model

    def build(self):
        """
        构建GRU + CTC模型
        :return:
        """
        self.gru_model, self.ctc_model = self._gru_ctc_init()

        print(self.gru_model.summary())
        print(self.ctc_model.summary())

        return self.gru_model, self.ctc_model
