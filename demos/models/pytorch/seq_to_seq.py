import torch
import logging
from torch.utils.tensorboard import SummaryWriter

from demos.models.pytorch.seq2seq.encoder import Encoder
from demos.models.pytorch.seq2seq.decoder import Decoder
from demos.models.pytorch.seq2seq.seq2seq import Seq2Seq
from demos.models.pytorch.seq2seq.optimizer import LasOptimizer


class SeqToSeq(object):
    def __init__(self,
                 LFR_m,
                 LFR_n,
                 einput,
                 ehidden,
                 elayer,
                 edropout,
                 ebidirectional,
                 etype,
                 atype,
                 dembed,
                 dhidden,
                 dlayer,
                 epochs,
                 half_lr,
                 early_stop,
                 max_norm,
                 batch_size,
                 maxlen_in,
                 maxlen_out,
                 num_workers,
                 optimizer,
                 lr,
                 momentum,
                 l2,
                 checkpoint):
        self.LFR_m = LFR_m
        self.LFR_n = LFR_n
        self.einput = einput
        self.ehidden = ehidden
        self.elayer = elayer
        self.edropout = edropout
        self.ebidirectional = ebidirectional
        self.etype = etype
        self.atype = atype
        self.dembed = dembed
        self.dhidden = dhidden
        self.dlayer = dlayer
        self.epochs = epochs
        self.half_lr = half_lr
        self.early_stop = early_stop
        self.max_norm = max_norm
        self.batch_size = batch_size
        self.maxlen_in = maxlen_in
        self.maxlen_out = maxlen_out
        self.num_workers = num_workers
        self.optimizer = optimizer
        self.lr = lr
        self.momentum = momentum
        self.l2 = l2
        self.checkpoint = checkpoint

        self.vocab_size = None
        self.sos_id = None
        self.eos_id = None
        self.pad_id = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    @staticmethod
    def get_logger():
        """
        log 日志
        :return:
        """
        logger = logging.getLogger()
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s %(levelname)s \t%(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

        return logger

    def train(self):
        """
        模型训练
        :return:
        """

        start_epoch = 0
        best_loss = float('inf')
        writer = SummaryWriter()
        epochs_since_improvement = 0

        # Initialize / load checkpoint
        if self.checkpoint is None:
            encoder = Encoder(input_size=self.einput * self.LFR_m,
                              hidden_size=self.ehidden,
                              num_layers=self.elayer,
                              dropout=self.edropout,
                              bidirectional=self.ebidirectional,
                              rnn_type=self.etype)
            decoder = Decoder(vocab_size=self.vocab_size,
                              embedding_dim=self.dembed,
                              sos_id=self.sos_id,
                              eos_id=self.eos_id,
                              pad_id=self.pad_id,
                              hidden_size=self.dhidden,
                              num_layers=self.dlayer,
                              bidirectional_encoder=self.ebidirectional)
            model = Seq2Seq(encoder, decoder)

            model.to(self.device)
            optimizer = LasOptimizer(
                torch.optim.Adam(model.parameters(), lr=self.lr, betas=(0.9, 0.98), eps=1e-09))
        else:
            checkpoint = torch.load(self.checkpoint)
            start_epoch = checkpoint['epoch'] + 1
            epochs_since_improvement = checkpoint['epochs_since_improvement']
            model = checkpoint['model']
            optimizer = checkpoint['optimizer']

        logger = SeqToSeq.get_logger()
