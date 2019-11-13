import torch
import logging
import tqdm
import numpy as np
from torch.utils.data.dataloader import default_collate
from torch.utils.tensorboard import SummaryWriter
from demos.utils.data_gen import AiShellDataset

from demos.utils.util import AverageMeter
from demos.models.pytorch.seq2seq.encoder import Encoder
from demos.models.pytorch.seq2seq.decoder import Decoder
from demos.models.pytorch.seq2seq.seq2seq import Seq2Seq
from demos.models.pytorch.seq2seq.optimizer import LasOptimizer

from configurations.constant import Constant


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
                 checkpoint,
                 print_freq,
                 vocab_size,
                 pad_id,
                 sos_id,
                 eos_id):
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
        self.print_freq = print_freq

        self.vocab_size = vocab_size
        self.pad_id = pad_id
        self.sos_id = sos_id
        self.eos_id = eos_id
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

    def pad_collate(self, batch):
        """

        :param batch:
        :return:
        """
        max_input_len = float('-inf')
        max_target_len = float('-inf')

        for elem in batch:
            feature, trn = elem
            max_input_len = max_input_len if max_input_len > feature.shape[0] else feature.shape[0]
            max_target_len = max_target_len if max_target_len > len(trn) else len(trn)

        for i, elem in enumerate(batch):
            feature, trn = elem
            input_length = feature.shape[0]
            input_dim = feature.shape[1]
            padded_input = np.zeros((max_input_len, input_dim), dtype=np.float32)
            padded_input[:input_length, :] = feature
            padded_target = np.pad(trn, (0, max_target_len - len(trn)), 'constant', constant_values=self.pad_id)
            batch[i] = (padded_input, padded_target, input_length)

        # sort it by input lengths (long to short)
        batch.sort(key=lambda x: x[2], reverse=True)

        return default_collate(batch)

    @staticmethod
    def save_checkpoint(epoch, epochs_since_improvement, model, optimizer, loss, is_best):
        state = {'epoch': epoch,
                 'epochs_since_improvement': epochs_since_improvement,
                 'loss': loss,
                 'model': model,
                 'optimizer': optimizer}

        filename = 'checkpoint.tar'
        torch.save(state, filename)
        # If this checkpoint is the best so far, store a copy so it doesn't get overwritten by a worse checkpoint
        if is_best:
            torch.save(state, 'BEST_checkpoint.tar')

    def train_net(self):
        """
        模型训练
        :return:
        """

        start_epoch = 0
        best_loss = float('inf')
        writer = SummaryWriter()
        epochs_since_improvement = 0

        # Initialize / load checkpoint
        if self.checkpoint == 'None':
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

        # Custom dataloaders
        train_dataset = AiShellDataset('train')
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size,
                                                   collate_fn=self.pad_collate, pin_memory=True,
                                                   shuffle=True, num_workers=self.num_workers)
        valid_dataset = AiShellDataset('dev')
        valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=self.batch_size,
                                                   collate_fn=self.pad_collate, pin_memory=True,
                                                   shuffle=False, num_workers=self.num_workers)

        # Epochs
        for epoch in range(start_epoch, self.epochs):
            print(self.epochs)
            # One epoch's training
            train_loss = self.train(train_loader=train_loader,
                                    model=model,
                                    optimizer=optimizer,
                                    epoch=epoch,
                                    logger=logger)
            writer.add_scalar('model/train_loss', train_loss, epoch)

            lr = optimizer.lr
            print('\nLearning rate: {}'.format(lr))
            step_num = optimizer.step_num
            print('Step num: {}\n'.format(step_num))

            writer.add_scalar('model/learning_rate', lr, epoch)

            # One epoch's validation
            valid_loss = self.valid(valid_loader=valid_loader,
                                    model=model,
                                    logger=logger)
            writer.add_scalar('model/valid_loss', valid_loss, epoch)

            # Check if there was an improvement
            is_best = valid_loss < best_loss
            best_loss = min(valid_loss, best_loss)
            if not is_best:
                epochs_since_improvement += 1
                print("\nEpochs since last improvement: %d\n" % (epochs_since_improvement,))
            else:
                epochs_since_improvement = 0

            # Save checkpoint
            SeqToSeq.save_checkpoint(epoch, epochs_since_improvement, model, optimizer, best_loss, is_best)

    def train(self, train_loader, model, optimizer, epoch, logger):
        model.train()  # train mode (dropout and batchnorm is used)

        losses = AverageMeter()

        # Batches
        for i, (data) in enumerate(train_loader):
            # Move to GPU, if available
            padded_input, padded_target, input_lengths = data
            padded_input = padded_input.to(self.device)
            padded_target = padded_target.to(self.device)
            input_lengths = input_lengths.to(self.device)

            # Forward prop.
            loss = model(padded_input, input_lengths, padded_target)

            # Back prop.
            optimizer.zero_grad()
            loss.backward()

            # Update weights
            optimizer.step()

            # Keep track of metrics
            losses.update(loss.item())

            # Print status
            if i % self.print_freq == 0:
                logger.info('Epoch: [{0}][{1}/{2}]\t'
                            'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(epoch, i, len(train_loader), loss=losses))

        return losses.avg

    def valid(self, valid_loader, model, logger):
        model.eval()

        losses = AverageMeter()

        # Batches
        for data in tqdm(valid_loader):
            # Move to GPU, if available
            padded_input, padded_target, input_lengths = data
            padded_input = padded_input.to(self.device)
            padded_target = padded_target.to(self.device)
            input_lengths = input_lengths.to(self.device)

            # Forward prop.
            loss = model(padded_input, input_lengths, padded_target)

            # Keep track of metrics
            losses.update(loss.item())

        # Print status
        logger.info('\nValidation Loss {loss.val:.4f} ({loss.avg:.4f})\n'.format(loss=losses))

        return losses.avg


if __name__ == '__main__':
    configuration = Constant().get_configuration()
    project_path = Constant().get_project_path()
    datasource_type = configuration['datasource_type']
    model_type = configuration['model_type']

    SeqToSeq(LFR_m=configuration[model_type]['LFR_m'],
             LFR_n=configuration[model_type]['LFR_n'],
             einput=configuration[model_type]['einput'],
             ehidden=configuration[model_type]['ehidden'],
             elayer=configuration[model_type]['elayer'],
             edropout=configuration[model_type]['edropout'],
             ebidirectional=configuration[model_type]['ebidirectional'],
             etype=configuration[model_type]['etype'],
             atype=configuration[model_type]['atype'],
             dembed=configuration[model_type]['dembed'],
             dhidden=configuration[model_type]['dhidden'],
             dlayer=configuration[model_type]['dlayer'],
             epochs=configuration[model_type]['epochs'],
             half_lr=configuration[model_type]['half_lr'],
             early_stop=configuration[model_type]['early_stop'],
             max_norm=configuration[model_type]['max_norm'],
             batch_size=configuration[model_type]['batch_size'],
             maxlen_in=configuration[model_type]['maxlen_in'],
             maxlen_out=configuration[model_type]['maxlen_out'],
             num_workers=configuration[model_type]['num_workers'],
             optimizer=configuration[model_type]['optimizer'],
             lr=configuration[model_type]['lr'],
             momentum=configuration[model_type]['momentum'],
             l2=configuration[model_type]['l2'],
             checkpoint=configuration[model_type]['checkpoint'],
             print_freq=configuration[model_type]['print_freq'],
             vocab_size=configuration[datasource_type]['vocab_size'],
             pad_id=configuration['token']['PAD'],
             sos_id=configuration['token']['SOS'],
             eos_id=configuration['token']['EOS']).train_net()
