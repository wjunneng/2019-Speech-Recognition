import argparse
import os
import pickle
from shutil import copyfile

import torch
from tqdm import tqdm
import numpy as np
import random

from configurations.constant import Constant
from demos.utils.data_gen import AiShellDataset
from demos.models.pytorch.seq2seq.seq2seq import Seq2Seq


class Test(object):
    def __init__(self):
        configuration = Constant().get_configuration()
        model_type = configuration['model_type']
        datasource_type = configuration['datasource_type']

        self.LFR_m = configuration[model_type]['LFR_m']
        self.LFR_n = configuration[model_type]['LFR_n']
        self.input_dim = configuration[datasource_type]['input_dim']
        self.sos_id = configuration['token']['SOS']
        self.eos_id = configuration['token']['EOS']

        self.device = torch.device('cuda:0' if torch.cuda.is_available() is True else 'cpu')
        self.beam_size = 5
        self.nbest = 1
        self.decode_max_len = 100

    @staticmethod
    def levenshtein(u, v):
        prev = None
        curr = [0] + list(range(1, len(v) + 1))
        # Operations: (SUB, DEL, INS)
        prev_ops = None
        curr_ops = [(0, 0, i) for i in range(len(v) + 1)]
        for x in range(1, len(u) + 1):
            prev, curr = curr, [x] + ([None] * len(v))
            prev_ops, curr_ops = curr_ops, [(0, x, 0)] + ([None] * len(v))
            for y in range(1, len(v) + 1):
                delcost = prev[y] + 1
                addcost = curr[y - 1] + 1
                subcost = prev[y - 1] + int(u[x - 1] != v[y - 1])
                curr[y] = min(subcost, delcost, addcost)
                if curr[y] == subcost:
                    (n_s, n_d, n_i) = prev_ops[y - 1]
                    curr_ops[y] = (n_s + int(u[x - 1] != v[y - 1]), n_d, n_i)
                elif curr[y] == delcost:
                    (n_s, n_d, n_i) = prev_ops[y]
                    curr_ops[y] = (n_s, n_d + 1, n_i)
                else:
                    (n_s, n_d, n_i) = curr_ops[y - 1]
                    curr_ops[y] = (n_s, n_d, n_i + 1)
        return curr[len(v)], curr_ops[len(v)]

    @staticmethod
    def cer_function(ref, hyp):
        wer_s, wer_i, wer_d, wer_n = 0, 0, 0, 0
        cer_s, cer_i, cer_d, cer_n = 0, 0, 0, 0
        sen_err = 0
        for n in range(len(ref)):
            # update CER statistics
            _, (s, i, d) = Test.levenshtein(ref[n], hyp[n])
            cer_s += s
            cer_i += i
            cer_d += d
            cer_n += len(ref[n])
            # update WER statistics
            _, (s, i, d) = Test.levenshtein(ref[n].split(), hyp[n].split())
            wer_s += s
            wer_i += i
            wer_d += d
            wer_n += len(ref[n].split())
            # update SER statistics
            if s + i + d > 0:
                sen_err += 1

        print(cer_s, cer_i, cer_d, cer_n)
        return (cer_s + cer_i + cer_d) / cer_n

    def main(self):
        # with open('/home/wjunneng/Ubuntu/Speech-Recognition/data/data_aishell/audio_index.pkl', 'rb') as file:
        #     data = pickle.load(file)
        # char_list = data['IVOCAB']
        # samples = data['train']
        #
        # checkpoint = '/home/wjunneng/Ubuntu/2019-Speech-Recognition/demos/models/pytorch/BEST_checkpoint.tar'
        # checkpoint = torch.load(checkpoint)
        # model = checkpoint['model']
        # model.eval()
        #
        # num_samples = len(samples)
        #
        # total_cer = 0
        #
        # for i in tqdm(range(num_samples)):
        #     sample = samples[i]
        #     wave = sample['wav_path']
        #     trn = sample['token_index']
        #
        #     feature = AiShellDataset.extract_feature(input_file=wave, feature='fbank', dim=self.input_dim, cmvn=True)
        #     feature = AiShellDataset.build_LFR_features(feature, m=self.LFR_m, n=self.LFR_n)
        #     # feature = np.expand_dims(feature, axis=0)
        #     input = torch.from_numpy(feature).to(self.device)
        #     input_length = [input.shape[0]]
        #     input_length = torch.from_numpy(np.array(input_length)).long().to(self.device)
        #
        #     with torch.no_grad():
        #         nbest_hyps = model.recognize(input=input, input_length=input_length, beam_size=self.beam_size,
        #                                      nbest=self.nbest, decode_max_len=self.decode_max_len)
        #
        #     hyp_list = []
        #     for hyp in nbest_hyps:
        #         out = hyp['yseq']
        #         out = [char_list[idx] for idx in out if idx not in (self.sos_id, self.eos_id)]
        #         out = ''.join(out)
        #         hyp_list.append(out)
        #
        #     print(hyp_list)
        #
        #     gt = [char_list[idx] for idx in trn if idx not in (self.sos_id, self.eos_id)]
        #     gt = ''.join(gt)
        #     gt_list = [gt]
        #
        #     print(gt_list)
        #
        #     cer = Test.cer_function(gt_list, hyp_list)
        #     total_cer += cer
        #
        # avg_cer = total_cer / num_samples
        #
        # print('Average CER: ' + str(avg_cer))

        with open('/home/wjunneng/Ubuntu/Speech-Recognition/data/data_aishell/audio_index.pkl', 'rb') as file:
            data = pickle.load(file)
        char_list = data['IVOCAB']
        samples = data['test']

        filename = 'listen-attend-spell.pt'
        model = Seq2Seq()
        model.load_state_dict(torch.load(filename))
        model = model.to(self.device)
        model.eval()

        samples = random.sample(samples, 10)
        if not os.path.isdir('audios'):
            os.mkdir('audios')

        results = []

        for i, sample in enumerate(samples):
            wave = sample['wave']
            trn = sample['trn']

            copyfile(wave, 'audios/audio_{}.wav'.format(i))

            input = AiShellDataset.extract_feature(input_file=wave, feature='fbank', dim=self.input_dim, cmvn=True)
            input = AiShellDataset.build_LFR_features(input, m=self.LFR_m, n=self.LFR_n)
            # print(input.shape)

            # input = np.expand_dims(input, axis=0)
            input = torch.from_numpy(input).to(self.device)
            input_length = [input.shape[0]]
            input_length = torch.LongTensor(input_length).to(self.device)

            with torch.no_grad():
                nbest_hyps = model.recognize(input=input, input_length=input_length, beam_size=self.beam_size,
                                             nbest=self.nbest, decode_max_len=self.decode_max_len)

            out_list = []
            for hyp in nbest_hyps:
                out = hyp['yseq']
                out = [char_list[idx] for idx in out]
                out = ''.join(out).replace('<sos>', '').replace('<eos>', '')
                out_list.append(out)
            out = out_list[0]
            print('OUT: {}'.format(out))

            gt = [char_list[idx] for idx in trn]
            gt = ''.join(gt).replace('<eos>', '')
            print(' GT: {}\n'.format(gt))

            results.append({'out_list_{}'.format(i): out_list, 'gt_{}'.format(i): gt})

        import json

        with open('results.json', 'w') as file:
            json.dump(results, file, indent=4, ensure_ascii=False)


if __name__ == '__main__':
    Test().main()
