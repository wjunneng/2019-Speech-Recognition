import os
import numpy as np
import torch
from torch import nn

from demos.utils import data_util, model_util


class SpeechRecognizer:
    def __init__(self,
                 char2ind,
                 ind2char,
                 save_path,
                 mode='TRAIN',
                 num_layers_encoder=1,
                 num_layers_decoder=1,
                 embedding_dim=300,
                 rnn_size_encoder=256,
                 rnn_size_decoder=256,
                 learning_rate=0.001,
                 learning_rate_decay=0.9,
                 learning_rate_decay_steps=100,
                 max_lr=0.01,
                 keep_probability_i=0.825,
                 keep_probability_o=0.895,
                 keep_probability_h=0.86,
                 keep_probability_e=0.986,
                 batch_size=64,
                 beam_width=10,
                 epochs=20,
                 eos="<eos>",
                 bos="<bos>",
                 pad='<pad>',
                 clip=5,
                 inference_targets=False,
                 summary_dir=None,
                 use_cyclic_lr=False):
        self.char2ind = char2ind
        self.ind2char = ind2char
        self.vocab_size = len(char2ind)
        self.num_layers_encoder = num_layers_encoder
        self.num_layers_decoder = num_layers_decoder
        self.rnn_size_encoder = rnn_size_encoder
        self.rnn_size_decoder = rnn_size_decoder
        self.save_path = save_path
        self.embedding_dim = embedding_dim
        self.mode = mode.upper()
        self.learning_rate = learning_rate
        self.learning_rate_decay = learning_rate_decay
        self.learning_rate_decay_steps = learning_rate_decay_steps
        self.keep_probability_i = keep_probability_i
        self.keep_probability_o = keep_probability_o
        self.keep_probability_h = keep_probability_h
        self.keep_probability_e = keep_probability_e
        self.batch_size = batch_size
        self.beam_width = beam_width
        self.eos = eos
        self.bos = bos
        self.clip = clip
        self.pad = pad
        self.epochs = epochs
        self.inference_targets = inference_targets
        self.use_cyclic_lr = use_cyclic_lr
        self.max_lr = max_lr
        self.summary_dir = summary_dir

    def build_graph(self):
        """
        构建动态图
        :return:
        """
        self.embedding = nn.Embedding(num_embeddings=self.vocab_size+1, embedding_dim=self.embedding_dim)
        self.char_embedding = self.embedding(self.char)