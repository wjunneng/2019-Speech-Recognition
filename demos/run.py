import os

from demos.models.tensorflow.speech_recognizer import SpeechRecognizer
from demos.utils import data_util
from demos.utils import model_util
from configurations.constant import Constant


class Run(object):
    def __init__(self):
        self.constant = Constant().get_configuration()
        self.model_path = os.path.join(Constant().get_project_path(), self.constant['path']['model_path'])

    def main(self):
        model_util.Util.reset_graph()

        data = data_util.Util()

        # audios_sorted, txts_sorted, audio_paths_sorted, txts_splitted_sorted, txts_converted_sorted =
        sr = SpeechRecognizer(char2ind=data.char2ind,
                              ind2char=data.ind2char,
                              save_path=self.model_path,
                              num_layers_encoder=self.constant['num_layers_encoder'],
                              num_layers_decoder=self.constant["num_layers_decoder"],
                              rnn_size_encoder=self.constant["rnn_size_encoder"],
                              rnn_size_decoder=self.constant["rnn_size_decoder"],
                              embedding_dim=self.constant["embedding_dim"],
                              batch_size=self.constant["batch_size"],
                              epochs=self.constant["epochs"],
                              use_cyclic_lr=self.constant["use_cyclic_lr"],
                              learning_rate=self.constant["learning_rate"],
                              max_lr=self.constant["max_lr"],
                              learning_rate_decay_steps=self.constant["learning_rate_decay_steps"])

        sr.build_graph()
        sr.train(data.audios[:100], data.txts_converted[:100], restore_path=self.model_path)

        model_util.Util.reset_graph()
        sr = SpeechRecognizer(char2ind=data.char2ind,
                              ind2char=data.ind2char,
                              save_path=self.model_path,
                              num_layers_encoder=self.constant["num_layers_encoder"],
                              num_layers_decoder=self.constant["num_layers_decoder"],
                              rnn_size_encoder=self.constant["rnn_size_encoder"],
                              rnn_size_decoder=self.constant["rnn_size_decoder"],
                              mode="INFER",
                              embedding_dim=self.constant["embedding_dim"],
                              batch_size=1,
                              beam_width=5)

        sr.build_graph()
        preds = sr.infer(data.audios[0:10:2], restore_path=self.model_path)

        preds_converted = data_util.Util().preds2txt(preds, data.ind2char, beam=True)

        data_util.Util().print_samples(preds_converted, data.txts_splitted[0:10:2])


if __name__ == '__main__':
    Run().main()
