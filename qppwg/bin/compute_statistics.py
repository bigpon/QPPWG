#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2020 Yi-Chiao Wu (Nagoya University)
# based on a Parallel WaveGAN script by Tomoki Hayashi (Nagoya University)
# (https://github.com/kan-bayashi/ParallelWaveGAN)
#  MIT License (https://opensource.org/licenses/MIT)

import argparse
import logging
import yaml

from sklearn.preprocessing import StandardScaler
from joblib import dump
from qppwg.utils import (read_txt, read_hdf5)


def calc_stats(file_list, config, shift=1):
    """Calcute statistics

    Args:
        file_list (list): File list.
        config (dict): Dictionary of config.
        shift (int): Shift of feature dimesion.

    """
    scaler = StandardScaler()

    # process over all of data
    for i, filename in enumerate(file_list):
        logging.info("now processing %s (%d/%d)" % (filename, i + 1, len(file_list)))
        feat = read_hdf5(filename, "/%s" % config['feat_type'])
        scaler.partial_fit(feat[:, shift:])

    dump(scaler, config['stats'])


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--feats", required=True, type=str,
                        help="name of the list of hdf5 files")
    parser.add_argument("--config", required=True, type=str,
                        help="yaml format configuration file")
    parser.add_argument("--stats", required=True, type=str,
                        help="filename of stats")
    parser.add_argument("--verbose", default=1, type=int,
                        help="log message level")

    args = parser.parse_args()

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

    # read file list
    file_list = read_txt(args.feats)
    logging.info("number of utterances = %d" % len(file_list))

    # load config and speaker info
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    config.update(vars(args))

    # calculate statistics
    shift = config.get("stats_shift", 1)
    # for world feature,
    # the first dimesion (u/v) is usually
    # skipped in calculating statistics.
    logging.info("stats shift dimesion: %d" % shift)
    calc_stats(file_list, config, shift)


if __name__ == "__main__":
    main()
