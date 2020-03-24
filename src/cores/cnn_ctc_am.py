# -*- coding:utf-8 -*-
import os
import sys

os.chdir(sys.path[0])
from tensorflow.contrib.keras import layers, models, backend


class CNNCTCAM(object):
    def __init__(self, args):
        self.OUTPUT_SIZE = args.vocab_size
        self.AUDIO_LENGTH = args.audio_length
        self.AUDIO_FEATURE_LENGTH = args.audio_length

    def _cnn_init(self):
        self.input_data = layers.Input(name='the input', shape=(self.AUDIO_LENGTH, self.AUDIO_FEATURE_LENGTH, 1))

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

        layers_h16 = layers.Reshape((self.AUDIO_FEATURE_LENGTH, self.AUDIO_LENGTH))(layers_h15)
        layers_h16 = layers.Dropout(rate=0.3)(layers_h16)

        layers_h17 = layers.Dense(units=128, use_bias=True, activation='relu', kernel_initializer='he_normal')(
            layers_h16)
        layers_h17 = layers.Dropout(rate=0.3)(layers_h17)

        layers_h18 = layers.Dense(units=self.OUTPUT_SIZE, use_bias=True, activation='relu',
                                  kernel_initializer='he_normal')(layers_h17)

        self.y_pred = layers.Activation('softmax', name='Activation0')(layers_h18)

        return models.Model(inputs=self.input_data, outputs=self.y_pred)

    def _ctc_init(self):
        self.labels = layers.Input(name='the label', shape=[None], dtype='float32')
        self.input_length = layers.Input(name='input length', shape=[1], dtype='int64')
        self.label_length = layers.Input(name='label length', shape=[1], dtype='int64')
        self.loss = layers.Lambda(
            backend.ctc_batch_cost(y_true=self.labels, y_pred=self.y_pred, input_length=self.input_length,
                                   label_length=self.label_length), output_shape=(1,), name='ctc')
        return models.Model(inputs=[self.labels, self.input_data, self.input_length, self.label_length],
                            outputs=self.loss)

    def build(self):
        """
        构建CNN + CTC模型
        :return:
        """
        self.cnn_model = self._cnn_init()
        self.ctc_model = self._ctc_init()