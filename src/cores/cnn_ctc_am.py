# -*- coding:utf-8 -*-
import os
import sys

os.chdir(sys.path[0])
from tensorflow.contrib.keras import layers, models, backend, optimizers


class CNNCTCAM(object):
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

    def _cnn_ctc_init(self):
        self.input_data = layers.Input(name='the_input', shape=(self.AUDIO_LENGTH, self.AUDIO_FEATURE_LENGTH, 1))

        layers_h1 = layers.Conv2D(filters=32, kernel_size=(3, 3), use_bias=False, activation='relu', padding='same',
                                  kernel_initializer='he_normal')(self.input_data)
        layers_h1 = layers.Dropout(rate=0.05)(layers_h1)

        layers_h2 = layers.Conv2D(filters=32, kernel_size=(3, 3), use_bias=True, activation='relu', padding='same',
                                  kernel_initializer='he_normal')(layers_h1)

        layers_h3 = layers.MaxPooling2D(pool_size=2, strides=None, padding='valid')(layers_h2)
        layers_h3 = layers.Dropout(rate=0.05)(layers_h3)

        layers_h4 = layers.Conv2D(filters=64, kernel_size=(3, 3), use_bias=True, activation='relu', padding='same',
                                  kernel_initializer='he_normal')(layers_h3)
        layers_h4 = layers.Dropout(rate=0.1)(layers_h4)

        layers_h5 = layers.Conv2D(filters=64, kernel_size=(3, 3), use_bias=True, activation='relu', padding='same',
                                  kernel_initializer='he_normal')(layers_h4)

        layers_h6 = layers.MaxPooling2D(pool_size=2, strides=None, padding='valid')(layers_h5)
        layers_h6 = layers.Dropout(rate=0.1)(layers_h6)

        layers_h7 = layers.Conv2D(filters=128, kernel_size=(3, 3), use_bias=True, activation='relu', padding='same',
                                  kernel_initializer='he_normal')(layers_h6)
        layers_h7 = layers.Dropout(rate=0.15)(layers_h7)

        layers_h8 = layers.Conv2D(filters=128, kernel_size=(3, 3), use_bias=True, activation='relu', padding='same',
                                  kernel_initializer='he_normal')(layers_h7)

        layers_h9 = layers.MaxPooling2D(pool_size=2, strides=None, padding='valid')(layers_h8)
        layers_h9 = layers.Dropout(rate=0.15)(layers_h9)

        layers_h10 = layers.Conv2D(filters=128, kernel_size=(3, 3), use_bias=True, activation='relu', padding='same',
                                   kernel_initializer='he_normal')(layers_h9)
        layers_h10 = layers.Dropout(rate=0.2)(layers_h10)

        layers_h11 = layers.Conv2D(filters=128, kernel_size=(3, 3), use_bias=True, activation='relu', padding='same',
                                   kernel_initializer='he_normal')(layers_h10)

        layers_h12 = layers.MaxPooling2D(pool_size=1, strides=None, padding='valid')(layers_h11)

        layers_h12 = layers.Dropout(rate=0.2)(layers_h12)

        layers_h13 = layers.Conv2D(filters=128, kernel_size=(3, 3), use_bias=True, activation='relu', padding='same',
                                   kernel_initializer='he_normal')(layers_h12)
        layers_h13 = layers.Dropout(rate=0.2)(layers_h13)

        layers_h14 = layers.Conv2D(filters=128, kernel_size=(3, 3), use_bias=True, activation='relu', padding='same',
                                   kernel_initializer='he_normal')(layers_h13)

        layers_h15 = layers.MaxPooling2D(pool_size=1, strides=None, padding='valid')(layers_h14)

        layers_h16 = layers.Reshape((self.AUDIO_FEATURE_LENGTH, self.AUDIO_LENGTH * 2))(layers_h15)
        layers_h16 = layers.Dropout(rate=0.3)(layers_h16)

        layers_h17 = layers.Dense(units=128, use_bias=True, activation='relu', kernel_initializer='he_normal')(
            layers_h16)
        layers_h17 = layers.Dropout(rate=0.3)(layers_h17)

        layers_h18 = layers.Dense(units=self.OUTPUT_SIZE, use_bias=True, kernel_initializer='he_normal')(layers_h17)

        y_pred = layers.Activation('softmax', name='activation_0')(layers_h18)

        self.cnn_model = models.Model(inputs=self.input_data, outputs=y_pred)

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

        return self.cnn_model, self.ctc_model

    def build(self):
        """
        构建CNN + CTC模型
        :return:
        """
        self.cnn_model, self.ctc_model = self._cnn_ctc_init()

        print(self.cnn_model.summary())
        print(self.ctc_model.summary())

        return self.cnn_model, self.ctc_model
