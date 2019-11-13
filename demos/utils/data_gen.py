import os
from torch.utils.data import Dataset
import pickle
import librosa
import random
import numpy as np

from demos.utils.pre_process import PreProcess
from configurations.constant import Constant


class AiShellDataset(Dataset):
    def __init__(self, split):
        self.configuration = Constant().get_configuration()
        self.datasource_type = self.configuration['datasource_type']
        self.model_type = self.configuration['model_type']
        self.audio_index_pkl_path = os.path.join(self.configuration[self.datasource_type]['path'],
                                                 self.configuration[self.datasource_type]['audio_index_pkl_path'])
        self.input_dim = self.configuration[self.datasource_type]['input_dim']
        self.LFR_m = self.configuration[self.model_type]['LFR_m']
        self.LFR_n = self.configuration[self.model_type]['LFR_n']

        with open(self.audio_index_pkl_path, 'rb') as file:
            data = pickle.load(file)

        # dev or train or test
        self.samples = data[split]
        print('loading {} {} samples...'.format(len(self.samples), split))

    @staticmethod
    def normalize(yt):
        yt_max = np.max(yt)
        yt_min = np.min(yt)
        a = 1.0 / (yt_max - yt_min)
        b = -(yt_max + yt_min) / (2 * (yt_max - yt_min))

        yt = yt * a + b

        return yt

    @staticmethod
    def extract_feature(input_file, feature='fbank', dim=80, cmvn=True, delta=False, delta_delta=False,
                        window_size=25, stride=10, save_feature=None):
        """
        # Acoustic Feature Extraction
        # Parameters
        #     - input file  : str, audio file path
        #     - feature     : str, fbank or mfcc
        #     - dim         : int, dimension of feature
        #     - cmvn        : bool, apply CMVN on feature
        #     - window_size : int, window size for FFT (ms)
        #     - stride      : int, window stride for FFT
        #     - save_feature: str, if given, store feature to the path and return len(feature)
        # Return
        # acoustic features with shape (time step, dim)

        :param input_file:
        :param feature:
        :param dim:
        :param cmvn:
        :param delta:
        :param delta_delta:
        :param window_size:
        :param stride:
        :param save_feature:
        :return:
        """
        y, sr = librosa.load(input_file, sr=None)
        yt, _ = librosa.effects.trim(y, top_db=20)
        yt = AiShellDataset.normalize(yt)
        ws = int(sr * 0.001 * window_size)
        st = int(sr * 0.001 * stride)
        if feature == 'fbank':  # log-scaled
            feat = librosa.feature.melspectrogram(y=yt, sr=sr, n_mels=dim,
                                                  n_fft=ws, hop_length=st)
            feat = np.log(feat + 1e-6)
        elif feature == 'mfcc':
            feat = librosa.feature.mfcc(y=yt, sr=sr, n_mfcc=dim, n_mels=26,
                                        n_fft=ws, hop_length=st)
            feat[0] = librosa.feature.rmse(yt, hop_length=st, frame_length=ws)

        else:
            raise ValueError('Unsupported Acoustic Feature: ' + feature)

        feat = [feat]
        if delta:
            feat.append(librosa.feature.delta(feat[0]))

        if delta_delta:
            feat.append(librosa.feature.delta(feat[0], order=2))
        feat = np.concatenate(feat, axis=0)
        if cmvn:
            feat = (feat - feat.mean(axis=1)[:, np.newaxis]) / (feat.std(axis=1) + 1e-16)[:, np.newaxis]
        if save_feature is not None:
            tmp = np.swapaxes(feat, 0, 1).astype('float32')
            np.save(save_feature, tmp)
            return len(tmp)
        else:
            return np.swapaxes(feat, 0, 1).astype('float32')

    @staticmethod
    def build_LFR_features(inputs, m, n):
        """
        Actually, this implements stacking frames and skipping frames.
        if m = 1 and n = 1, just return the origin features.
        if m = 1 and n > 1, it works like skipping.
        if m > 1 and n = 1, it works like stacking but only support right frames.
        if m > 1 and n > 1, it works like LFR.
        Args:
            inputs_batch: inputs is T x D np.ndarray
            m: number of frames to stack
            n: number of frames to skip
        """
        # LFR_inputs_batch = []
        # for inputs in inputs_batch:
        LFR_inputs = []
        T = inputs.shape[0]
        T_lfr = int(np.ceil(T / n))
        for i in range(T_lfr):
            if m <= T - i * n:
                LFR_inputs.append(np.hstack(inputs[i * n:i * n + m]))
            else:  # process last LFR frame
                num_padding = m - (T - i * n)
                frame = np.hstack(inputs[i * n:])
                for _ in range(num_padding):
                    frame = np.hstack((frame, inputs[-1]))
                LFR_inputs.append(frame)

        return np.vstack(LFR_inputs)

    @staticmethod
    def spec_augment(spec: np.ndarray, num_mask=2, freq_masking=0.15, time_masking=0.20, value=0):
        """
        Source: https://www.kaggle.com/davids1992/specaugment-quick-implementation
        提取spec 特征
        :param spec:
        :param num_mask:
        :param freq_masking:
        :param time_masking:
        :param value:
        :return:
        """
        spec = spec.copy()
        num_mask = random.randint(1, num_mask)
        for i in range(num_mask):
            all_freqs_num, all_frames_num = spec.shape
            freq_percentage = random.uniform(0.0, freq_masking)

            num_freqs_to_mask = int(freq_percentage * all_freqs_num)
            f0 = np.random.uniform(low=0.0, high=all_freqs_num - num_freqs_to_mask)
            f0 = int(f0)
            spec[f0:f0 + num_freqs_to_mask, :] = value

            time_percentage = random.uniform(0.0, time_masking)

            num_frames_to_mask = int(time_percentage * all_frames_num)
            t0 = np.random.uniform(low=0.0, high=all_frames_num - num_frames_to_mask)
            t0 = int(t0)
            spec[:, t0:t0 + num_frames_to_mask] = value
        return spec

    def __getitem__(self, item):
        sample = self.samples[item]
        wav_path = sample['wav_path']
        token_index = sample['token_index']

        feature = AiShellDataset.extract_feature(input_file=wav_path,
                                                 feature='fbank',
                                                 dim=self.input_dim)
        feature = AiShellDataset.build_LFR_features(feature, m=self.LFR_m, n=self.LFR_n)
        # zero mean and unit variance
        feature = (feature - feature.mean()) / feature.std()
        feature = AiShellDataset.spec_augment(feature)

        return feature, token_index

    def __len__(self):
        return len(self.samples)
