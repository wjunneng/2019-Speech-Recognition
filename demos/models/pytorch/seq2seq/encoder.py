import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class Encoder(nn.Module):
    def __init__(self, input_size=320, hidden_size=256, num_layers=3,
                 dropout=0.0, bidirectional=True, rnn_type='lstm'):
        super(Encoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.rnn_type = rnn_type
        self.lstm = nn.LSTM(input_size=self.input_size,
                            hidden_size=self.hidden_size,
                            batch_first=True,
                            dropout=self.dropout,
                            bidirectional=self.bidirectional)

    def forward(self, padding_input, input_lengths):
        """
        Args:
            padded_input: N x T x D
            input_lengths: N  保存每条序列的长度

        Returns: output, hidden
            - **output**: N x T x H
            - **hidden**: (num_layers * num_directions) x N x H
        """
        # Add total_length for supportting nn.DataParallel() later
        # see https://pytorch.org/docs/stable/notes/faq.html#pack-rnn-unpack-with-data-parallelism
        # T
        total_length = padding_input.size(1)
        packed_input = pack_padded_sequence(input=padding_input, lengths=input_lengths, batch_first=True)
        packed_output, hidden = self.lstm(packed_input)
        output, _ = pad_packed_sequence(sequence=packed_output, batch_first=True, total_length=total_length)

        return output, hidden
