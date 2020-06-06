#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2020 Yi-Chiao Wu (Nagoya University)
# based on a Parallel WaveGAN script by Tomoki Hayashi (Nagoya University)
# (https://github.com/kan-bayashi/ParallelWaveGAN)
#  MIT License (https://opensource.org/licenses/MIT)

"""Decode with trained Quasi-Periodic Parallel WaveGAN Generator."""

import argparse
import logging
import os
import time
import numpy as np
import soundfile as sf
import torch
import yaml

from tqdm import tqdm
import qppwg.models
from qppwg.datasets import FeatDataset
from qppwg.utils import read_hdf5


def main():
    """Run decoding process."""
    parser = argparse.ArgumentParser(
        description="Decode dumped features with trained Quasi-Periodic Parallel WaveGAN Generator "
                    "(See detail in qppwg/bin/decode.py).")
    parser.add_argument("--eval_feat", required=True, type=str,
                        help="list of evaluation aux feat files")
    parser.add_argument("--stats", required=True, type=str,
                        help="hdf5 file including statistics")
    parser.add_argument("--indir", required=True, type=str,
                        help="directory of input feature files")
    parser.add_argument("--outdir", type=str, required=True,
                        help="directory to output generated speech.")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="checkpoint file to be loaded.")
    parser.add_argument("--config", default=None, type=str,
                        help="yaml format configuration file. if not explicitly provided, "
                             "it will be searched in the checkpoint directory. (default=None)")
    parser.add_argument("--verbose", type=int, default=1,
                        help="logging level. higher is more logging. (default=1)")
    parser.add_argument("--seed", default=100, type=int,
                        help="seed number")
    parser.add_argument("--f0_factor", default=1.0, type=float,
                        help="f0 scaled factor")
    args = parser.parse_args()

    # set logger
    if args.verbose > 1:
        logging.basicConfig(
            level=logging.DEBUG, format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s")
    elif args.verbose > 0:
        logging.basicConfig(
            level=logging.INFO, format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s")
    else:
        logging.basicConfig(
            level=logging.WARN, format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s")
        logging.warning("Skip DEBUG/INFO messages")

    # fix seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)

    # check directory existence
    if not os.path.isdir(os.path.dirname(args.outdir)):
        os.makedirs(os.path.dirname(args.outdir))

    # load config
    if args.config is None:
        dirname = os.path.dirname(args.checkpoint)
        args.config = os.path.join(dirname, "config.yml")
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.Loader)
    config.update(vars(args))

    # get dataset
    feat_load_fn = lambda x: read_hdf5(x, config.get("feat_type", "world"))
    f0_factor = args.f0_factor
    dataset = FeatDataset(
        stats=args.stats,
        feat_list=args.eval_feat,
        feat_load_fn=feat_load_fn,
        return_filename=True,
        hop_size=config["hop_size"],
        dense_factor=config.get("dense_factor", 4),
        f0_threshold=config.get("f0_threshold", 0),
        f0_cont=config.get("f0_cont", True),
        f0_dim_idx=config.get("f0_dim_idx", 1),
        uv_dim_idx=config.get("uv_dim_idx", 0),
        mean_path=config.get("mean_path", "/world/mean"),
        scale_path=config.get("scale_path", "/world/scale"),
        f0_factor=f0_factor,
        fs=config.get("sampling_rate", 22050),
        shift=config.get("stats_shift", 1),
    )
    logging.info(f"The number of features to be decoded = {len(dataset)}.")

    # setup
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    model_class = getattr(
        qppwg.models,
        config.get("generator_type", "ParallelWaveGANGenerator"))
    model = model_class(**config["generator_params"])
    model.load_state_dict(
        torch.load(args.checkpoint, map_location="cpu")["model"]["generator"])
    logging.info(f"Loaded model parameters from {args.checkpoint}.")
    model.remove_weight_norm()
    model = model.eval().to(device)
    input_type = config.get("input_type", "noise")
    pad_fn = torch.nn.ReplicationPad1d(
        config["generator_params"].get("aux_context_window", 0))

    # start generation
    total_rtf = 0.0
    with torch.no_grad(), tqdm(dataset, desc="[decode]") as pbar:
        for idx, (feat_path, c, d) in enumerate(pbar, 1):
            # setup input
            x = ()
            if input_type == "noise":
                z = torch.randn(1, 1, len(c) * config["hop_size"]).to(device)
                x += (z,)
            else:
                raise NotImplementedError("Currently only 'noise' input is supported ")
            c = pad_fn(torch.FloatTensor(c).unsqueeze(0).transpose(2, 1)).to(device)
            d = torch.FloatTensor(d).view(1, 1, -1).to(device)
            x += (c, d,)

            # generate
            start = time.time()
            y = model(*x).view(-1).cpu().numpy()
            rtf = (time.time() - start) / (len(y) / config["sampling_rate"])
            pbar.set_postfix({"RTF": rtf})
            total_rtf += rtf

            # save as PCM 16 bit wav file
            feat_path = os.path.splitext(feat_path)[0]
            feat_path = feat_path.replace(args.indir, args.outdir)
            if f0_factor == 1.0:  # unchanged
                wav_filename = "%s.wav" % (feat_path)
            else:  # scaled f0
                wav_filename = "%s_f%.2f.wav" % (feat_path, f0_factor)
            if not os.path.exists(os.path.dirname(wav_filename)):
                os.makedirs(os.path.dirname(wav_filename))
            sf.write(wav_filename, y, config.get("sampling_rate", 22050), "PCM_16")

    # report average RTF
    logging.info(f"Finished generation of {idx} utterances (RTF = {total_rtf / idx:.03f}).")


if __name__ == "__main__":
    main()
