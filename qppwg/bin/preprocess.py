#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2020 Wu Yi-Chiao (Nagoya University)
# based on a Parallel WaveGAN script by Tomoki Hayashi (Nagoya University)
# (https://github.com/kan-bayashi/ParallelWaveGAN)
# based on sprocket-vc script by Kazuhiro Kobayashi (Nagoya University)
# (https://github.com/k2kobayashi/sprocket)
#  MIT License (https://opensource.org/licenses/MIT)

from __future__ import division

import argparse
import logging
import multiprocessing as mp
import os
import sys
import copy
import yaml
import pyworld
import librosa
import numpy as np
import soundfile as sf

from distutils.util import strtobool
from scipy.interpolate import interp1d
from scipy.io import wavfile
from scipy.signal import firwin
from scipy.signal import lfilter
from sprocket.speech import FeatureExtractor
from sprocket.speech import Synthesizer
from qppwg.utils import (read_txt, read_hdf5, write_hdf5, check_hdf5)


def _get_arguments():
    parser = argparse.ArgumentParser(
        description="making feature file argsurations.")
    # setting
    parser.add_argument("--audio", required=True,
                        type=str, help="List of input wav files")
    parser.add_argument("--indir", required=True,
                        type=str, help="Directory of input feature files")
    parser.add_argument("--outdir", required=True,
                        type=str, help="Directory to save generated samples")
    parser.add_argument("--config", required=True,
                        type=str, help="YAML format configuration file")
    parser.add_argument("--spkinfo", default="None",
                        type=str, help="YAML format speaker information")
    parser.add_argument("--feature_format", default="h5",
                        type=str, help="Feature format")
    parser.add_argument("--split", default="/",
                        type=str, help="Path split string")
    parser.add_argument("--spkidx", default=-2,
                        type=int, help="Speaker index of the split path")
    # flags setting
    parser.add_argument("--save_f0", default=True,
                        type=strtobool, help="If set True, features f0 will be saved")
    parser.add_argument("--save_ap", default=False,
                        type=strtobool, help="If set True, features ap will be saved")
    parser.add_argument("--save_spc", default=False,
                        type=strtobool, help="If set True, features spc will be saved")
    parser.add_argument("--save_npow", default=True,
                        type=strtobool, help="If set True, features npow will be saved")
    # other setting
    parser.add_argument('--inv', default=True,
                        type=strtobool, help="If False, wav is restored from acoustic features")
    parser.add_argument("--verbose", default=1,
                        type=int, help="Log message level")

    return parser.parse_args()


def path_create(wav_list, indir, outdir, extname):
    for wav_name in wav_list:
        path_replace(wav_name, indir, outdir, extname=extname)


def path_replace(filepath, inputpath, outputpath, extname=None):
    filepath = filepath.replace(inputpath, outputpath)
    if not os.path.exists(os.path.dirname(filepath)):
        os.makedirs(os.path.dirname(filepath))
    if extname is not None:
        filepath = '%s.%s' % (filepath.split('.')[0], extname)
    return filepath


def spk_division(file_list, config, spkinfo, split="/", spkidx=-2):
    """Divide list into speaker-dependent list

    Args:
        file_list (list): Waveform list
        config (dict): Config
        spkinfo (dict): Dictionary of
            speaker-dependent f0 range and power threshold
        split: Path split string
        spkidx: Speaker index of the split path
    Return:
        (list): List of divided file lists
        (list): List of speaker-dependent configs

    """
    file_lists, configs, tempf = [], [], []
    prespk = None
    for file in file_list:
        spk = file.split(split)[spkidx]
        if spk != prespk:
            if tempf:
                file_lists.append(tempf)
            tempf = []
            prespk = spk
            tempc = copy.deepcopy(config)
            if spk in spkinfo:
                tempc['minf0']  = spkinfo[spk]['f0_min']
                tempc['maxf0']  = spkinfo[spk]['f0_max']
                tempc['pow_th'] = spkinfo[spk]['pow_th']
            else:
                msg = "Since %s is not in spkinfo dict, " % spk
                msg += "default f0 range and power threshold are used."
                logging.info(msg)
                tempc['minf0']  = 40
                tempc['maxf0']  = 800
                tempc['pow_th'] = -20
            configs.append(tempc)
        tempf.append(file)
    file_lists.append(tempf)

    return file_lists, configs


def aux_list_create(wav_list_file, config):
    """Create list of auxiliary acoustic features

    Args:
        wav_list_file (str): Filename of wav list
        config (dict): Config

    """
    aux_list_file = wav_list_file.replace(".scp", ".list")
    wav_files = read_txt(wav_list_file)
    with open(aux_list_file, "w") as f:
        for wav_name in wav_files:
            feat_name = path_replace(wav_name,
                                     config['indir'], config['outdir'],
                                     extname=config['feature_format'])
            f.write("%s\n" % feat_name)


def low_cut_filter(x, fs, cutoff=70):
    """Low cut filter

    Args:
        x (ndarray): Waveform sequence
        fs (int): Sampling frequency
        cutoff (float): Cutoff frequency of low cut filter
    Return:
        (ndarray): Low cut filtered waveform sequence

    """
    nyquist = fs // 2
    norm_cutoff = cutoff / nyquist
    fil = firwin(255, norm_cutoff, pass_zero=False)
    lcf_x = lfilter(fil, 1, x)

    return lcf_x


def low_pass_filter(x, fs, cutoff=70, padding=True):
    """Low pass filter

    Args:
        x (ndarray): Waveform sequence
        fs (int): Sampling frequency
        cutoff (float): Cutoff frequency of low pass filter
    Return:
        (ndarray): Low pass filtered waveform sequence

    """
    nyquist = fs // 2
    norm_cutoff = cutoff / nyquist
    numtaps = 255
    fil = firwin(numtaps, norm_cutoff)
    x_pad = np.pad(x, (numtaps, numtaps), 'edge')
    lpf_x = lfilter(fil, 1, x_pad)
    lpf_x = lpf_x[numtaps + numtaps // 2: -numtaps // 2]

    return lpf_x


# WORLD features
def convert_continuos_f0(f0):
    """Convert F0 to continuous F0

    Args:
        f0 (ndarray): original f0 sequence with the shape (T)
    Return:
        (ndarray): continuous f0 with the shape (T)

    """
    # get uv information as binary
    uv = np.float32(f0 != 0)
    # get start and end of f0
    if (f0 == 0).all():
        logging.warn("all of the f0 values are 0.")
        return uv, f0
    start_f0 = f0[f0 != 0][0]
    end_f0 = f0[f0 != 0][-1]
    # padding start and end of f0 sequence
    cont_f0 = copy.deepcopy(f0)
    start_idx = np.where(cont_f0 == start_f0)[0][0]
    end_idx = np.where(cont_f0 == end_f0)[0][-1]
    cont_f0[:start_idx] = start_f0
    cont_f0[end_idx:] = end_f0
    # get non-zero frame index
    nz_frames = np.where(cont_f0 != 0)[0]
    # perform linear interpolation
    f = interp1d(nz_frames, cont_f0[nz_frames])
    cont_f0 = f(np.arange(0, cont_f0.shape[0]))

    return uv, cont_f0


def world_speech_synthesis(queue, wav_list, config):
    """WORLD speech synthesis

    Args:
        queue (multiprocessing.Queue): the queue to store the file name of utterance
        wav_list (list): list of the wav files
        config (dict): feature extraction config

    """
    # define synthesizer
    synthesizer = Synthesizer(fs=config['sampling_rate'],
                              fftl=config['fft_size'],
                              shiftms=config['shiftms'])
    # synthesis
    for i, wav_name in enumerate(wav_list):
        logging.info("now processing %s (%d/%d)" % (wav_name, i + 1, len(wav_list)))

        # load acoustic features
        feat_name = path_replace(wav_name, config['indir'],
                                 config['outdir'], extname=config['feature_format'])
        if check_hdf5(feat_name, "/world"):
            h = read_hdf5(feat_name, "/world")
        else:
            logging.error("%s is not existed." % (feat_name))
            sys.exit(1)
        if check_hdf5(feat_name, "/f0"):
            f0 = read_hdf5(feat_name, "/f0")
        else:
            uv = h[:, config['uv_dim_idx']].copy(order='C')
            f0 = h[:, config['f0_dim_idx']].copy(order='C')  # cont_f0_lpf
            fz_idx = np.where(uv == 0.0)
            f0[fz_idx] = 0.0
        if check_hdf5(feat_name, "/ap"):
            ap = read_hdf5(feat_name, "/ap")
        else:
            codeap = h[:, config['ap_dim_start']:config['ap_dim_end']].copy(order='C')
            ap = pyworld.decode_aperiodicity(codeap, config['sampling_rate'], config['fft_size'])
        mcep = h[:, config['mcep_dim_start']:config['mcep_dim_end']].copy(order='C')

        # waveform synthesis
        wav = synthesizer.synthesis(f0,
                                    mcep,
                                    ap,
                                    alpha=config['mcep_alpha'])
        wav = np.clip(np.int16(wav), -32768, 32767)

        # save restored wav
        restored_name = path_replace(wav_name, "wav", "world", extname="wav")
        wavfile.write(restored_name, config['sampling_rate'], wav)

    queue.put('Finish')


def world_feature_extract(queue, wav_list, config):
    """WORLD feature extraction

    Args:
        queue (multiprocessing.Queue): the queue to store the file name of utterance
        wav_list (list): list of the wav files
        config (dict): feature extraction config

    """
    # define feature extractor
    feature_extractor = FeatureExtractor(
        analyzer="world",
        fs=config['sampling_rate'],
        shiftms=config['shiftms'],
        minf0=config['minf0'],
        maxf0=config['maxf0'],
        fftl=config['fft_size'])
    # extraction
    for i, wav_name in enumerate(wav_list):
        logging.info("now processing %s (%d/%d)" % (wav_name, i + 1, len(wav_list)))

        # load wavfile and apply low cut filter
        fs, x = wavfile.read(wav_name)
        x = np.array(x, dtype=np.float32)
        if config['highpass_cutoff'] != 0:
            x = low_cut_filter(x, fs, cutoff=config['highpass_cutoff'])

        # check sampling frequency
        if not fs == config['sampling_rate']:
            logging.error("sampling frequency of %s is not matched." % wav_name)
            sys.exit(1)

        # extract features
        f0, spc, ap = feature_extractor.analyze(x)
        codeap = feature_extractor.codeap()
        mcep = feature_extractor.mcep(dim=config['mcep_dim'], alpha=config['mcep_alpha'])
        npow = feature_extractor.npow()
        uv, cont_f0 = convert_continuos_f0(f0)
        lpf_fs = int(1.0 / (config['shiftms'] * 0.001))
        cont_f0_lpf = low_pass_filter(cont_f0, lpf_fs, cutoff=20)
        next_cutoff = 70
        while not (cont_f0_lpf >= [0]).all():
            cont_f0_lpf = low_pass_filter(cont_f0, lpf_fs, cutoff=next_cutoff)
            next_cutoff *= 2

        # concatenate
        cont_f0_lpf = np.expand_dims(cont_f0_lpf, axis=-1)
        uv = np.expand_dims(uv, axis=-1)
        feats = np.concatenate([uv, cont_f0_lpf, mcep, codeap], axis=1)

        # save feature
        feat_name = path_replace(wav_name, config['indir'],
                                 config['outdir'], extname=config['feature_format'])
        write_hdf5(feat_name, "/%s" % (config["feat_type"]), feats)
        if config['save_f0']:
            write_hdf5(feat_name, "/f0", f0)
        if config['save_ap']:
            write_hdf5(feat_name, "/ap", ap)
        if config['save_spc']:
            write_hdf5(feat_name, "/spc", spc)
        if config['save_npow']:
            write_hdf5(feat_name, "/npow", npow)

    queue.put('Finish')


# Mel-spec and f0 features
def logmelfilterbank(audio,
                     sampling_rate,
                     fft_size=1024,
                     hop_size=256,
                     win_length=None,
                     window="hann",
                     num_mels=80,
                     fmin=None,
                     fmax=None,
                     eps=1e-10,
                     ):
    """Extract log-Mel filterbank feature.

    Args:
        audio (ndarray): Audio signal (T,).
        sampling_rate (int): Sampling rate.
        fft_size (int): FFT size.
        hop_size (int): Hop size.
        win_length (int): Window length. If set to None, it will be the same as fft_size.
        window (str): Window function type.
        num_mels (int): Number of mel basis.
        fmin (int): Minimum frequency in mel basis calculation.
        fmax (int): Maximum frequency in mel basis calculation.
        eps (float): Epsilon value to avoid inf in log calculation.

    Returns:
        ndarray: Log Mel filterbank feature (#frames, num_mels).

    """
    # get amplitude spectrogram
    x_stft = librosa.stft(audio, n_fft=fft_size, hop_length=hop_size,
                          win_length=win_length, window=window, pad_mode="reflect")
    spc = np.abs(x_stft).T  # (#frames, #bins)

    # get mel basis
    fmin = 0 if fmin is None else fmin
    fmax = sampling_rate / 2 if fmax is None else fmax
    mel_basis = librosa.filters.mel(sampling_rate, fft_size, num_mels, fmin, fmax)

    return np.log10(np.maximum(eps, np.dot(spc, mel_basis.T)))


def melf0_feature_extract(queue, wav_list, config):
    """Mel-spc w/ F0 feature extraction

    Args:
        queue (multiprocessing.Queue): the queue to store the file name of utterance
        wav_list (list): list of the wav files
        config (dict): feature extraction config

    """
    # define f0 feature extractor
    feature_extractor = FeatureExtractor(
        analyzer="world",
        fs=config['sampling_rate'],
        shiftms=config['shiftms'],
        minf0=config['minf0'],
        maxf0=config['maxf0'],
        fftl=config['fft_size'])
    # extraction
    for i, wav_name in enumerate(wav_list):
        logging.info("now processing %s (%d/%d)" % (wav_name, i + 1, len(wav_list)))

        # load wavfile
        (x, fs) = sf.read(wav_name)

        # check sampling frequency
        if not fs == config['sampling_rate']:
            logging.error("sampling frequency is not matched.")
            sys.exit(1)

        # extract f0 and uv features
        f0, _, _ = feature_extractor.analyze(x)
        uv, cont_f0 = convert_continuos_f0(f0)
        lpf_fs = int(1.0 / (config['shiftms'] * 0.001))
        cont_f0_lpf = low_pass_filter(cont_f0, lpf_fs, cutoff=20)
        next_cutoff = 70
        while not (cont_f0_lpf >= [0]).all():
            cont_f0_lpf = low_pass_filter(cont_f0, lpf_fs, cutoff=next_cutoff)
            next_cutoff *= 2

        # extract mel-spc feature
        mel = logmelfilterbank(x, fs,
                               fft_size=config["fft_size"],
                               hop_size=config["hop_size"],
                               win_length=config["win_length"],
                               window=config["window"],
                               num_mels=config["num_mels"],
                               fmin=config["fmin"],
                               fmax=config["fmax"])

        # concatenate
        cont_f0_lpf = np.expand_dims(cont_f0_lpf, axis=-1)
        uv = np.expand_dims(uv, axis=-1)
        minlen = min(uv.shape[0], mel.shape[0])
        feats = np.concatenate([uv[:minlen, :], cont_f0_lpf[:minlen, :],
                                mel.astype(np.float32)[:minlen, :]], axis=1)

        # save feature
        feat_name = path_replace(wav_name, config['indir'],
                                 config['outdir'], extname=config['feature_format'])
        write_hdf5(feat_name, "/%s" % (config["feat_type"]), feats)
        if config['save_f0']:
            write_hdf5(feat_name, "/f0", f0)

    queue.put('Finish')


def main():
    # parser arguments
    args = _get_arguments()
    # set log level
    if args.verbose == 1:
        logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s',
                            datefmt='%m/%d/%Y %I:%M:%S')
    elif args.verbose > 1:
        logging.basicConfig(level=logging.DEBUG,
                            format='%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s',
                            datefmt='%m/%d/%Y %I:%M:%S')
    else:
        logging.basicConfig(level=logging.WARN,
                            format='%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s',
                            datefmt='%m/%d/%Y %I:%M:%S')
        logging.warn("logging is disabled.")

    # show argument
    for key, value in vars(args).items():
        logging.info("%s = %s" % (key, str(value)))

    # read list
    file_list = read_txt(args.audio)
    logging.info("number of utterances = %d" % len(file_list))

    # load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
        config.update(vars(args))

    # list division
    if os.path.exists(args.spkinfo):
        # load speaker info
        with open(args.spkinfo, "r") as f:
            spkinfo = yaml.safe_load(f)
        # divide into each spk list
        file_lists, configs = spk_division(file_list, config, spkinfo)
    else:
        msg = "Since spkinfo %s is not exist, " % args.spkinfo
        msg += "default f0 range and power threshold are used."
        logging.info(msg)
        file_lists = np.array_split(file_list, 10)
        file_lists = [f_list.tolist() for f_list in file_lists]
        config['minf0']  = 40
        config['maxf0']  = 800
        config['pow_th'] = -20
        configs = [config] * len(file_lists)

    # set mode
    if config['feat_type'] == "world":
        if args.inv:
            target_fn = world_feature_extract
            # create auxiliary feature list
            aux_list_create(args.audio, config)
            # create folder
            path_create(file_list, config['indir'],
                        config['outdir'], config['feature_format'])
        else:
            target_fn = world_speech_synthesis
            # create folder
            path_create(file_list, "wav", "world", "wav")
    elif config['feat_type'][:6] == "melf0h":
        if args.inv:
            target_fn = melf0_feature_extract
            # create auxiliary feature list
            aux_list_create(args.audio, config)
            # create folder
            path_create(file_list, config['indir'],
                        config['outdir'], config['feature_format'])
        else:
            raise NotImplementedError("Currently, only mel-spec extraction is supported.")
    else:
        raise NotImplementedError("Currently, only 'world' and 'melf0hxxx' are supported.")

    # multi processing
    processes = []
    queue = mp.Queue()
    for f, config in zip(file_lists, configs):
        p = mp.Process(target=target_fn, args=(queue, f, config,))
        p.start()
        processes.append(p)

    # wait for all process
    for p in processes:
        p.join()


if __name__ == "__main__":
    main()
