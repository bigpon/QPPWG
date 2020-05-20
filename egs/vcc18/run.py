#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2020 Yi-Chiao Wu (Nagoya University)
#  MIT License (https://opensource.org/licenses/MIT)

"""Pytorch QPPWG script

Usage: run.py [-h] [-g GPUID]
            [-f FACTOR] [-C CONFIG]
            [-T TRAIN] [-V VALID] [-E EVAL]
            [-R RESUME] [-I ITER]
            [-0] [-1] [-2]

Options:
    -h, --help   Show the help
    -g GPUID     GPU device ID
    -f FACTOR    F0 scaled factor
    -C CONFIG    Name of config version
    -T TRAIN     Training set
    -V VALID     Validation set
    -E EVAL      Evaluation set
    -R RESUME    Number of iteration to resume model
    -I ITER      Number of iteration of testing model
    -0, --step0  Execute step0 (Feature extraction)
    -1, --step1  Execute step1 (QPPWG training)
    -2, --step2  Execute step2 (QPPWG decodeing)

"""
import os
from docopt import docopt


# PATH INITIALIZATION
def _path_initial(pathlist):
    for pathdir in pathlist:
        if not os.path.exists(pathdir):
            os.makedirs(pathdir)


# PATH CHECK
def _path_check(pathlist):
    for pathdir in pathlist:
        if not os.path.exists(pathdir):
            raise FileNotFoundError("%s doesn't exist!!" % pathdir)


# PATH & PARAMETER SETTINGS
LIBRARY_DIR = "/usr/local/cuda-10.0/lib64"
CUDA_DIR    = "/usr/local/cuda-10.0"
PRJ_ROOT    = "../.."
SEED        = 1
DECODE_SEED = 100

# MAIN
if __name__ == "__main__":
    args = docopt(__doc__)
    print(args)
    # step control
    execute_steps = [args["--step{}".format(step_index)] for step_index in range(0, 3)]
    if not any(execute_steps):
        raise("Please specify steps with options")
    # environment setting
    os.environ['LD_LIBRARY_PATH'] += ":" + LIBRARY_DIR
    os.environ['CUDA_HOME'] = CUDA_DIR
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    if args['-g'] is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = args['-g']
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    # path setting
    network        = "qppwg"
    entry_fe       = "qppwg-preprocess"
    entry_stats    = "qppwg-compute-statistics"
    entry_train    = "qppwg-train"
    entry_decode   = "qppwg-decode"
    train_version  = "vcc18_train_22kHz"  # training
    valid_version  = "vcc18_valid_22kHz"  # validation
    eval_version   = "vcc18_eval_22kHz"   # evaluation
    config_version = "QPPWGaf_20"  # config
    model_iters    = "400000"  # iteration of testing model
    f0_factor      = "1.00"  # scaled factor of f0
    if args['-f'] is not None:
        f0_factor = args['-f']
    if args['-T'] is not None:
        train_version = args['-T']
    if args['-V'] is not None:
        valid_version = args['-V']
    if args['-E'] is not None:
        eval_version = args['-E']
    if args['-C'] is not None:
        config_version = args['-C']
    if args['-I'] is not None:
        model_iters = args['-I']
    model_version = "%s_%s" % (network, train_version)  # model name
    spkinfo       = "data/pow_f0_dict.yml"
    config        = "conf/vcc18.%s.yaml" % (config_version)
    stats         = "data/stats/%s.joblib" % (train_version)
    outdir        = "exp/%s_%s" % (model_version, config_version)
    train_wav     = "data/scp/%s.scp" % (train_version)
    valid_wav     = "data/scp/%s.scp" % (valid_version)
    eval_wav      = "data/scp/%s.scp" % (eval_version)
    train_aux     = "data/scp/%s.list" % (train_version)
    valid_aux     = "data/scp/%s.list" % (valid_version)
    eval_aux      = "data/scp/%s.list" % (eval_version)
    _path_check([config])

    # FEATURE EXTRACTION
    if execute_steps[0]:
        inverse = True  # If False, wav is restored from acoustic features
        split = "/"  # Path split string
        spkidx = -2  # Speaker index of the split path
        # feature extraction
        for wav in [train_wav, valid_wav, eval_wav]:
            _path_check([wav])
            cmd = entry_fe    + \
                " --audio "   + wav + \
                " --indir "   + "wav" + \
                " --outdir "  + "hdf5" + \
                " --config "  + config + \
                " --spkinfo " + spkinfo + \
                " --split "   + split + \
                " --spkidx "  + str(spkidx) + \
                " --inv "     + str(inverse) + \
                " --verbose 1 "
            os.system(cmd)
        # calculate statistic
        _path_check([train_aux])
        cmd = entry_stats + \
            " --feats "   + train_aux + \
            " --config "  + config + \
            " --stats "   + stats
        os.system(cmd)

    # NETWORK TRAINING
    if execute_steps[1]:
        # resume setting
        if args['-R'] is not None:
            resume = "%s/checkpoint-%ssteps.pkl" % (outdir, args['-R'])
        else:
            resume = "None"
        # training
        cmd = entry_train     + \
            " --train_audio " + train_wav + \
            " --train_feat "  + train_aux + \
            " --valid_audio " + valid_wav + \
            " --valid_feat "  + valid_aux + \
            " --stats "       + stats + \
            " --outdir "      + outdir + \
            " --config "      + config + \
            " --resume "      + resume + \
            " --seed "        + str(SEED) + \
            " --verbose 1 "
        os.system(cmd)

    # EVALUATION (ANALYSIS-SYNTHESIS)
    if execute_steps[2]:
        # path setting
        indir = "data/hdf5/"  # input path of features
        outdir_eval = "%s/wav/%s/" % (outdir, model_iters)  # wav output path
        # check trained model
        checkpoint = "%s/checkpoint-%ssteps.pkl" % (outdir, model_iters)
        _path_check([checkpoint])

        # speech decoding
        cmd = entry_decode   + \
            " --eval_feat "  + eval_aux + \
            " --stats "      + stats + \
            " --indir "      + indir + \
            " --outdir "     + outdir_eval + \
            " --checkpoint " + checkpoint + \
            " --config "     + config + \
            " --seed "       + str(DECODE_SEED) + \
            " --f0_factor "  + f0_factor + \
            " --verbose 1 "
        os.system(cmd)
