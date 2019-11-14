import torch.nn as nn

from demos.models.pytorch.seq2seq.decoder import Decoder
from demos.models.pytorch.seq2seq.encoder import Encoder


class Seq2Seq(nn.Module):
    """
    Sequence-to-Sequence architecture with configurable encoder and decoder.
    """

    def __init__(self, encoder=None, decoder=None):
        super(Seq2Seq, self).__init__()
        if encoder is not None and decoder is not None:
            self.encoder = encoder
            self.decoder = decoder
        else:
            self.encoder = Encoder()
            self.decoder = Decoder()

    def forward(self, padded_input, input_lengths, padded_target):
        """

        :param padded_input: N x Ti x D
        :param input_lengths: N
        :param padded_target: N x To
        :return:
        """
        encoder_padded_outputs, _ = self.encoder(padding_input=padded_input, input_lengths=input_lengths)
        loss = self.decoder(padded_input=padded_target, encoder_padded_outputs=encoder_padded_outputs)

        return loss

    def recognize(self, input, input_length, beam_size, nbest, decode_max_len):
        """
        Sequence-to-Sequence beam search, decode one utterence [发声] now.
        :param input: T x D
        :param input_length:
        :param beam_size:
        :param nbest:
        :param decode_max_len:
        :return:
        """
        encoder_outputs, _ = self.encoder(padding_input=input.unsqueeze(0), input_lengths=input_length)
        nbest_hyps = self.decoder.recognize_beam(encoder_outputs=encoder_outputs[0], beam_size=beam_size, nbest=nbest,
                                                 decode_max_len=decode_max_len)

        return nbest_hyps
