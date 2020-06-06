# -*- coding: utf-8 -*-

# Copyright 2020 Yi-Chiao Wu (Nagoya University)
# based on a Parallel WaveGAN script by Tomoki Hayashi (Nagoya University)
# (https://github.com/kan-bayashi/ParallelWaveGAN)
#  MIT License (https://opensource.org/licenses/MIT)

"""Dataset modules."""

import logging
import numpy as np

from multiprocessing import Manager
from torch.utils.data import Dataset
from qppwg.utils import read_hdf5, read_txt, check_filename
from qppwg.utils import validate_length, batch_f0, dilated_factor
from joblib import load

import soundfile as sf


class AudioFeatDataset(Dataset):
    """PyTorch compatible audio and acoustic feat. dataset."""

    def __init__(self,
                 stats,
                 audio_list,
                 feat_list,
                 audio_load_fn=sf.read,
                 feat_load_fn=lambda x: read_hdf5(x, "world"),
                 audio_length_threshold=None,
                 feat_length_threshold=None,
                 return_filename=False,
                 allow_cache=False,
                 hop_size=110,
                 dense_factor=4,
                 f0_threshold=0,
                 f0_cont=True,
                 f0_dim_idx=1,
                 uv_dim_idx=0,
                 mean_path="/world/mean",
                 scale_path="/world/scale",
                 shift=1,
                 ):
        """Initialize dataset.

        Args:
            stats (str): Filename of the statistic hdf5 file.
            audio_list (str): Filename of the list of audio files.
            feat_list (str): Filename of the list of feature files.
            audio_load_fn (func): Function to load audio file.
            feat_load_fn (func): Function to load feature file.
            audio_length_threshold (int): Threshold to remove short audio files.
            feat_length_threshold (int): Threshold to remove short feature files.
            return_filename (bool): Whether to return the filename with arrays.
            allow_cache (bool): Whether to allow cache of the loaded files.
            hop_size (int): Hope size of acoustic feature
            dense_factor (int): Number of taps in one cycle.
            f0_threshold (float): Lower bound of pitch.
            f0_cont (bool): Whether to get dilated factor by continuous f0.
            f0_dim_idx (int): Dimension index of f0. (if set -1, all dilated factors will be 1)
            uv_dim_idx (int): Dimension index of U/V.
            mean_path (str): The data path (channel) of the mean in the statistic hdf5 file.
            scale_path (str): The data path (channel) of the scale in the statistic hdf5 file.
            shift (int): Shift of feature dimesion.

        """
        # load audio and feature files & check filename
        audio_files = read_txt(audio_list)
        feat_files = read_txt(feat_list)
        assert check_filename(audio_files, feat_files)

        # filter by threshold
        if audio_length_threshold is not None:
            audio_lengths = [audio_load_fn(f).shape[0] for f in audio_files]
            idxs = [idx for idx in range(len(audio_files)) if audio_lengths[idx] > audio_length_threshold]
            if len(audio_files) != len(idxs):
                logging.warning(f"Some files are filtered by audio length threshold "
                                f"({len(audio_files)} -> {len(idxs)}).")
            audio_files = [audio_files[idx] for idx in idxs]
            feat_files = [feat_files[idx] for idx in idxs]
        if feat_length_threshold is not None:
            mel_lengths = [feat_load_fn(f).shape[0] for f in feat_files]
            idxs = [idx for idx in range(len(feat_files)) if mel_lengths[idx] > feat_length_threshold]
            if len(feat_files) != len(idxs):
                logging.warning(f"Some files are filtered by mel length threshold "
                                f"({len(feat_files)} -> {len(idxs)}).")
            audio_files = [audio_files[idx] for idx in idxs]
            feat_files = [feat_files[idx] for idx in idxs]

        # assert the number of files
        assert len(audio_files) != 0, f"${audio_list} is empty."
        assert len(audio_files) == len(feat_files), \
            f"Number of audio and mel files are different ({len(audio_files)} vs {len(feat_files)})."

        self.audio_files = audio_files
        self.audio_load_fn = audio_load_fn
        self.feat_load_fn = feat_load_fn
        self.feat_files = feat_files
        self.return_filename = return_filename
        self.allow_cache = allow_cache
        self.hop_size = hop_size
        self.f0_threshold = f0_threshold
        self.dense_factor = dense_factor
        self.f0_cont = f0_cont
        self.f0_dim_idx = f0_dim_idx
        self.uv_dim_idx = uv_dim_idx
        self.shift = shift

        if allow_cache:
            # NOTE(kan-bayashi): Manager is need to share memory in dataloader with num_workers > 0
            self.manager = Manager()
            self.caches = self.manager.list()
            self.caches += [() for _ in range(len(audio_files))]

        # define feature pre-processing function
        scaler = load(stats)
        self.feat_transform = lambda x: scaler.transform(x)

    def __getitem__(self, idx):
        """Get specified idx items.

        Args:
            idx (int): Index of the item.

        Returns:
            str: Utterance id (only in return_filename = True).
            ndarray: Audio signal (T,).
            ndarray: Feature (T', C).
            ndarray: Dilated factor (T, 1).

        """
        if self.allow_cache and len(self.caches[idx]) != 0:
            return self.caches[idx]

        audio, fs = self.audio_load_fn(self.audio_files[idx])
        feat = self.feat_load_fn(self.feat_files[idx])
        audio, feat = validate_length(audio, feat, self.hop_size)
        # get dilated factor sequence
        f0 = batch_f0(feat, self.f0_threshold, self.f0_cont, 
                      self.f0_dim_idx, self.uv_dim_idx)
        df = dilated_factor(f0, fs, self.dense_factor)
        df = df.repeat(self.hop_size, axis=0)
        # audio & feature pre-processing
        audio = audio.astype(np.float32)
        feat[:, self.shift:] = self.feat_transform(feat[:, self.shift:])

        if self.return_filename:
            items = self.feat_files[idx], audio, feat, df
        else:
            items = audio, feat, df

        if self.allow_cache:
            self.caches[idx] = items

        return items

    def __len__(self):
        """Return dataset length.

        Returns:
            int: The length of dataset.

        """
        return len(self.audio_files)


class FeatDataset(Dataset):
    """PyTorch compatible mel dataset."""

    def __init__(self,
                 stats,
                 feat_list,
                 feat_length_threshold=None,
                 feat_load_fn=lambda x: read_hdf5(x, "world"),
                 return_filename=False,
                 allow_cache=False,
                 hop_size=110,
                 dense_factor=4,
                 f0_threshold=0,
                 f0_cont=True,
                 f0_dim_idx=1,
                 uv_dim_idx=0,
                 mean_path="/world/mean",
                 scale_path="/world/scale",
                 f0_factor=1.0,
                 fs=22050,
                 shift=1,
                 ):
        """Initialize dataset.

        Args:
            stats (str): Filename of the statistic hdf5 file.
            feat_list (str): Filename of the list of feature files.
            feat_load_fn (func): Function to load feature file.
            feat_length_threshold (int): Threshold to remove short feature files.
            return_filename (bool): Whether to return the utterance id with arrays.
            allow_cache (bool): Whether to allow cache of the loaded files.
            hop_size (int): Hope size of acoustic feature
            dense_factor (int): Number of taps in one cycle.
            f0_threshold (float): Lower bound of pitch.
            f0_cont (bool): Whether to get dilated factor by continuous f0.
            f0_dim_idx (int): Dimension index of f0. (if set -1, all dilated factors will be 1)
            uv_dim_idx (int): Dimension index of U/V.
            mean_path (str): The data path (channel) of the mean in the statistic hdf5 file.
            scale_path (str): The data path (channel) of the scale in the statistic hdf5 file.
            f0_factor (float): Ratio of scaled f0
            fs (int): The sampling rate of audio speech
            shift (int): Shift of feature dimesion.

        """
        # load feat. files
        feat_files = read_txt(feat_list)

        # filter by threshold
        if feat_length_threshold is not None:
            mel_lengths = [feat_load_fn(f).shape[0] for f in feat_files]
            idxs = [idx for idx in range(len(feat_files)) if mel_lengths[idx] > feat_length_threshold]
            if len(feat_files) != len(idxs):
                logging.warning(f"Some files are filtered by mel length threshold "
                                f"({len(feat_files)} -> {len(idxs)}).")
            feat_files = [feat_files[idx] for idx in idxs]

        # assert the number of files
        assert len(feat_files) != 0, f"${feat_list} is empty."

        self.feat_files = feat_files
        self.feat_load_fn = feat_load_fn
        self.return_filename = return_filename
        self.allow_cache = allow_cache
        self.hop_size = hop_size
        self.dense_factor = dense_factor
        self.f0_threshold = f0_threshold
        self.f0_cont = f0_cont
        self.f0_factor = f0_factor
        self.f0_dim_idx = f0_dim_idx
        self.uv_dim_idx = uv_dim_idx
        self.fs = fs
        self.shift = shift

        if allow_cache:
            # NOTE(kan-bayashi): Manager is need to share memory in dataloader with num_workers > 0
            self.manager = Manager()
            self.caches = self.manager.list()
            self.caches += [() for _ in range(len(feat_files))]
        
        # define feature pre-processing function
        scaler = load(stats)
        self.feat_transform = lambda x: scaler.transform(x)

    def __getitem__(self, idx):
        """Get specified idx items.

        Args:
            idx (int): Index of the item.

        Returns:
            str: Utterance id (only in return_filename = True).
            ndarray: Feature (T', C).

        """
        if self.allow_cache and len(self.caches[idx]) != 0:
            return self.caches[idx]

        feat = self.feat_load_fn(self.feat_files[idx])
        # f0 scaled
        if self.f0_factor != 1.0:
            feat[:, self.f0_dim_idx] *= self.f0_factor  
        # get dilated factor sequence
        f0 = batch_f0(feat, self.f0_threshold, self.f0_cont, 
                      self.f0_dim_idx, self.uv_dim_idx)
        df = dilated_factor(f0, self.fs, self.dense_factor)
        df = df.repeat(self.hop_size, axis=0)
        # feature pre-processing
        feat[:, self.shift:] = self.feat_transform(feat[:, self.shift:])

        if self.return_filename:
            items = self.feat_files[idx], feat, df
        else:
            items = feat, df

        if self.allow_cache:
            self.caches[idx] = items

        return items

    def __len__(self):
        """Return dataset length.

        Returns:
            int: The length of dataset.

        """
        return len(self.feat_files)
