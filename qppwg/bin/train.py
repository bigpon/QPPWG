#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2020 Yi-Chiao Wu (Nagoya University)
# based on a Parallel WaveGAN script by Tomoki Hayashi (Nagoya University)
# (https://github.com/kan-bayashi/ParallelWaveGAN)
#  MIT License (https://opensource.org/licenses/MIT)

"""Train Quasi-Periodic Parallel WaveGAN."""

import argparse
import logging
import os
import sys

from collections import defaultdict

import matplotlib
import numpy as np
import soundfile as sf
import torch
import yaml

from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm

import qppwg
import qppwg.models
import qppwg.optimizers

from qppwg.datasets import AudioFeatDataset
from qppwg.losses import MultiResolutionSTFTLoss
from qppwg.utils import read_hdf5

# set to avoid matplotlib error in CLI environment
matplotlib.use("Agg")


class Trainer(object):
    """Customized trainer module for Quasi-Periodic Parallel WaveGAN training."""

    def __init__(self,
                 steps,
                 epochs,
                 data_loader,
                 model,
                 criterion,
                 optimizer,
                 scheduler,
                 config,
                 device=torch.device("cpu"),
                 ):
        """Initialize trainer.

        Args:
            steps (int): Initial global steps.
            epochs (int): Initial global epochs.
            data_loader (dict): Dict of data loaders. It must contrain "train" and "dev" loaders.
            model (dict): Dict of models. It must contrain "generator" and "discriminator" models.
            criterion (dict): Dict of criterions. It must contrain "stft" and "mse" criterions.
            optimizer (dict): Dict of optimizers. It must contrain "generator" and "discriminator" optimizers.
            scheduler (dict): Dict of schedulers. It must contrain "generator" and "discriminator" schedulers.
            config (dict): Config dict loaded from yaml format configuration file.
            device (torch.deive): Pytorch device instance.

        """
        self.steps = steps
        self.epochs = epochs
        self.data_loader = data_loader
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config
        self.device = device
        self.writer = SummaryWriter(config["outdir"])
        self.finish_train = False
        self.total_train_loss = defaultdict(float)
        self.total_eval_loss = defaultdict(float)

    def run(self):
        """Run training."""
        self.tqdm = tqdm(initial=self.steps,
                         total=self.config["train_max_steps"],
                         desc="[train]")
        while True:
            # train one epoch
            self._train_epoch()

            # check whether training is finished
            if self.finish_train:
                break

        self.tqdm.close()
        logging.info("Finished training.")

    def save_checkpoint(self, checkpoint_path):
        """Save checkpoint.

        Args:
            checkpoint_path (str): Checkpoint path to be saved.

        """
        state_dict = {
            "optimizer": {
                "generator": self.optimizer["generator"].state_dict(),
                "discriminator": self.optimizer["discriminator"].state_dict(),
            },
            "scheduler": {
                "generator": self.scheduler["generator"].state_dict(),
                "discriminator": self.scheduler["discriminator"].state_dict(),
            },
            "steps": self.steps,
            "epochs": self.epochs,
        }
        if self.config["distributed"]:
            state_dict["model"] = {
                "generator": self.model["generator"].module.state_dict(),
                "discriminator": self.model["discriminator"].module.state_dict(),
            }
        else:
            state_dict["model"] = {
                "generator": self.model["generator"].state_dict(),
                "discriminator": self.model["discriminator"].state_dict(),
            }

        if not os.path.exists(os.path.dirname(checkpoint_path)):
            os.makedirs(os.path.dirname(checkpoint_path))
        torch.save(state_dict, checkpoint_path)

    def load_checkpoint(self, checkpoint_path, load_only_params=False):
        """Load checkpoint.

        Args:
            checkpoint_path (str): Checkpoint path to be loaded.
            load_only_params (bool): Whether to load only model parameters.

        """
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        if self.config["distributed"]:
            self.model["generator"].module.load_state_dict(state_dict["model"]["generator"])
            self.model["discriminator"].module.load_state_dict(state_dict["model"]["discriminator"])
        else:
            self.model["generator"].load_state_dict(state_dict["model"]["generator"])
            self.model["discriminator"].load_state_dict(state_dict["model"]["discriminator"])
        if not load_only_params:
            self.steps = state_dict["steps"]
            self.epochs = state_dict["epochs"]
            self.optimizer["generator"].load_state_dict(state_dict["optimizer"]["generator"])
            self.optimizer["discriminator"].load_state_dict(state_dict["optimizer"]["discriminator"])
            # overwrite schedular argument parameters
            state_dict["scheduler"]["generator"].update(**self.config["generator_scheduler_params"])
            state_dict["scheduler"]["discriminator"].update(**self.config["discriminator_scheduler_params"])
            self.scheduler["generator"].load_state_dict(state_dict["scheduler"]["generator"])
            self.scheduler["discriminator"].load_state_dict(state_dict["scheduler"]["discriminator"])

    def _train_step(self, batch):
        """Train model one step."""
        # parse batch
        x, y = batch
        x = tuple([x_.to(self.device) for x_ in x])
        y = y.to(self.device)

        #######################
        #      Generator      #
        #######################
        # calculate generator loss
        y_ = self.model["generator"](*x)
        y, y_ = y.squeeze(1), y_.squeeze(1)
        sc_loss, mag_loss = self.criterion["stft"](y_, y)
        gen_loss = sc_loss + mag_loss
        if self.steps > self.config["discriminator_train_start_steps"]:
            gen_loss *= self.config.get("lambda_aux_after_introduce_adv_loss", 1.0)
            p_ = self.model["discriminator"](y_.unsqueeze(1))
            # for standard discriminator
            adv_loss = self.criterion["mse"](p_, p_.new_ones(p_.size()))
            self.total_train_loss["train/adversarial_loss"] += adv_loss.item()
            gen_loss += self.config["lambda_adv"] * adv_loss

        self.total_train_loss["train/spectral_convergence_loss"] += sc_loss.item()
        self.total_train_loss["train/log_stft_magnitude_loss"] += mag_loss.item()
        self.total_train_loss["train/generator_loss"] += gen_loss.item()

        # update generator
        self.optimizer["generator"].zero_grad()
        gen_loss.backward()
        if self.config["generator_grad_norm"] > 0:
            torch.nn.utils.clip_grad_norm_(
                self.model["generator"].parameters(),
                self.config["generator_grad_norm"])
        self.optimizer["generator"].step()
        self.scheduler["generator"].step()

        #######################
        #    Discriminator    #
        #######################
        if self.steps > self.config["discriminator_train_start_steps"]:
            # re-compute y_
            with torch.no_grad():
                y_ = self.model["generator"](*x)
            # calculate discriminator loss
            p = self.model["discriminator"](y.unsqueeze(1))
            p_ = self.model["discriminator"](y_.detach())
            # for standard discriminator
            real_loss = self.criterion["mse"](p, p.new_ones(p.size()))
            fake_loss = self.criterion["mse"](p_, p_.new_zeros(p_.size()))
            dis_loss = real_loss + fake_loss
            self.total_train_loss["train/real_loss"] += real_loss.item()
            self.total_train_loss["train/fake_loss"] += fake_loss.item()
            self.total_train_loss["train/discriminator_loss"] += dis_loss.item()

            # update discriminator
            self.optimizer["discriminator"].zero_grad()
            dis_loss.backward()
            if self.config["discriminator_grad_norm"] > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model["discriminator"].parameters(),
                    self.config["discriminator_grad_norm"])
            self.optimizer["discriminator"].step()
            self.scheduler["discriminator"].step()

        # update counts
        self.steps += 1
        self.tqdm.update(1)
        self._check_train_finish()

    def _train_epoch(self):
        """Train model one epoch."""
        for train_steps_per_epoch, batch in enumerate(self.data_loader["train"], 1):
            # train one step
            self._train_step(batch)

            # check interval
            if self.config["rank"] == 0:
                self._check_log_interval()
                self._check_eval_interval()
                self._check_save_interval()

            # check whether training is finished
            if self.finish_train:
                return

        # update
        self.epochs += 1
        self.train_steps_per_epoch = train_steps_per_epoch
        logging.info(f"(Steps: {self.steps}) Finished {self.epochs} epoch training "
                     f"({self.train_steps_per_epoch} steps per epoch).")

    @torch.no_grad()
    def _eval_step(self, batch):
        """Evaluate model one step."""
        # parse batch
        x, y = batch
        x = tuple([x_.to(self.device) for x_ in x])
        y = y.to(self.device)

        #######################
        #      Generator      #
        #######################
        y_ = self.model["generator"](*x)
        p_ = self.model["discriminator"](y_)
        y, y_ = y.squeeze(1), y_.squeeze(1)
        sc_loss, mag_loss = self.criterion["stft"](y_, y)
        aux_loss = sc_loss + mag_loss
        if self.steps > self.config["discriminator_train_start_steps"]:
            # keep compatibility
            aux_loss *= self.config.get("lambda_aux_after_introduce_adv_loss", 1.0)
        # for standard discriminator
        adv_loss = self.criterion["mse"](p_, p_.new_ones(p_.size()))
        gen_loss = aux_loss + self.config["lambda_adv"] * adv_loss

        #######################
        #    Discriminator    #
        #######################
        p = self.model["discriminator"](y.unsqueeze(1))
        p_ = self.model["discriminator"](y_.unsqueeze(1))
        # for standard discriminator
        real_loss = self.criterion["mse"](p, p.new_ones(p.size()))
        fake_loss = self.criterion["mse"](p_, p_.new_zeros(p_.size()))
        dis_loss = real_loss + fake_loss

        # add to total eval loss
        self.total_eval_loss["eval/adversarial_loss"] += adv_loss.item()
        self.total_eval_loss["eval/spectral_convergence_loss"] += sc_loss.item()
        self.total_eval_loss["eval/log_stft_magnitude_loss"] += mag_loss.item()
        self.total_eval_loss["eval/generator_loss"] += gen_loss.item()
        self.total_eval_loss["eval/real_loss"] += real_loss.item()
        self.total_eval_loss["eval/fake_loss"] += fake_loss.item()
        self.total_eval_loss["eval/discriminator_loss"] += dis_loss.item()

    def _eval_epoch(self):
        """Evaluate model one epoch."""
        logging.info(f"(Steps: {self.steps}) Start evaluation.")
        # change mode
        for key in self.model.keys():
            self.model[key].eval()

        # calculate loss for each batch
        for eval_steps_per_epoch, batch in enumerate(tqdm(self.data_loader["valid"], desc="[eval]"), 1):
            # eval one step
            self._eval_step(batch)

            # save intermediate result
            if eval_steps_per_epoch == 1:
                self._genearete_and_save_intermediate_result(batch)

        logging.info(f"(Steps: {self.steps}) Finished evaluation "
                     f"({eval_steps_per_epoch} steps per epoch).")

        # average loss
        for key in self.total_eval_loss.keys():
            self.total_eval_loss[key] /= eval_steps_per_epoch
            logging.info(f"(Steps: {self.steps}) {key} = {self.total_eval_loss[key]:.4f}.")

        # record
        self._write_to_tensorboard(self.total_eval_loss)

        # reset
        self.total_eval_loss = defaultdict(float)

        # restore mode
        for key in self.model.keys():
            self.model[key].train()

    @torch.no_grad()
    def _genearete_and_save_intermediate_result(self, batch):
        """Generate and save intermediate result."""
        # delayed import to avoid error related backend error
        import matplotlib.pyplot as plt

        # generate
        x_batch, y_batch = batch
        x_batch = tuple([x.to(self.device) for x in x_batch])
        y_batch = y_batch.to(self.device)
        y_batch_ = self.model["generator"](*x_batch)

        # check directory
        dirname = os.path.join(self.config["outdir"], f"predictions/{self.steps}steps")
        if not os.path.exists(dirname):
            os.makedirs(dirname)

        for idx, (y, y_) in enumerate(zip(y_batch, y_batch_), 1):
            # convert to ndarray
            y, y_ = y.view(-1).cpu().numpy(), y_.view(-1).cpu().numpy()

            # plot figure and save it
            figname = os.path.join(dirname, f"{idx}.png")
            plt.subplot(2, 1, 1)
            plt.plot(y)
            plt.title("groundtruth speech")
            plt.subplot(2, 1, 2)
            plt.plot(y_)
            plt.title(f"generated speech @ {self.steps} steps")
            plt.tight_layout()
            plt.savefig(figname)
            plt.close()

            # save as wavfile
            y = np.clip(y, -1, 1)
            y_ = np.clip(y_, -1, 1)
            sf.write(figname.replace(".png", "_ref.wav"), y,
                     self.config["sampling_rate"], "PCM_16")
            sf.write(figname.replace(".png", "_gen.wav"), y_,
                     self.config["sampling_rate"], "PCM_16")

            if idx >= self.config["num_save_intermediate_results"]:
                break

    def _write_to_tensorboard(self, loss):
        """Write to tensorboard."""
        for key, value in loss.items():
            self.writer.add_scalar(key, value, self.steps)

    def _check_save_interval(self):
        if self.steps % self.config["save_interval_steps"] == 0:
            self.save_checkpoint(
                os.path.join(self.config["outdir"], f"checkpoint-{self.steps}steps.pkl"))
            logging.info(f"Successfully saved checkpoint @ {self.steps} steps.")

    def _check_eval_interval(self):
        if self.steps % self.config["eval_interval_steps"] == 0:
            self._eval_epoch()

    def _check_log_interval(self):
        if self.steps % self.config["log_interval_steps"] == 0:
            for key in self.total_train_loss.keys():
                self.total_train_loss[key] /= self.config["log_interval_steps"]
                logging.info(f"(Steps: {self.steps}) {key} = {self.total_train_loss[key]:.4f}.")
            self._write_to_tensorboard(self.total_train_loss)

            # reset
            self.total_train_loss = defaultdict(float)

    def _check_train_finish(self):
        if self.steps >= self.config["train_max_steps"]:
            self.finish_train = True


class Collater(object):
    """Customized collater for Pytorch DataLoader in training."""

    def __init__(self,
                 batch_max_steps=25520,
                 hop_size=110,
                 aux_context_window=2,
                 input_type="noise",
                 ):
        """Initialize customized collater for PyTorch DataLoader.

        Args:
            batch_max_steps (int): The maximum length of input signal in batch.
            hop_size (int): Hop size of auxiliary features.
            aux_context_window (int): Context window size for auxiliary feature conv.
            input_type (str): The data type of input (currently only "noise" supported).

        """
        if batch_max_steps % hop_size != 0:
            batch_max_steps += -(batch_max_steps % hop_size)
        assert batch_max_steps % hop_size == 0
        self.batch_max_steps = batch_max_steps
        self.batch_max_frames = batch_max_steps // hop_size
        self.hop_size = hop_size
        self.aux_context_window = aux_context_window
        self.input_type = input_type

    def __call__(self, batch):
        """Convert into batch tensors.

        Args:
            batch (list): list of tuple of the pair of audio and features.

        Returns:
            Tensor: Gaussian noise batch (B, 1, T).
            Tensor: Auxiliary feature batch (B, C, T'), where T = (T' - 2 * aux_context_window) * hop_size
            Tensor: Target signal batch (B, 1, T).

        """
        # time resolution check
        y_batch, c_batch, d_batch = [], [], []
        for idx in range(len(batch)):
            x, c, d = batch[idx]
            self._check_length(x, c, d, self.hop_size, 0)
            if len(c) - 2 * self.aux_context_window > self.batch_max_frames:
                # randomly pickup with the batch_max_steps length of the part
                interval_start = self.aux_context_window
                interval_end = len(c) - self.batch_max_frames - self.aux_context_window
                start_frame = np.random.randint(interval_start, interval_end)
                start_step = start_frame * self.hop_size
                y = x[start_step: start_step + self.batch_max_steps]
                c = c[start_frame - self.aux_context_window:
                      start_frame + self.aux_context_window + self.batch_max_frames]
                d = d[start_step: start_step + self.batch_max_steps]
                self._check_length(y, c, d, self.hop_size, self.aux_context_window)
            else:
                logging.warn(f"Removed short sample from batch (length={len(x)}).")
                continue
            y_batch += [y.astype(np.float32).reshape(-1, 1)]  # [(T, 1), (T, 1), ...]
            c_batch += [c.astype(np.float32)]  # [(T' C), (T' C), ...]
            d_batch += [d.astype(np.float32).reshape(-1, 1)]  # [(T, 1), (T, 1), ...]

        # convert each batch to tensor, asuume that each item in batch has the same length
        y_batch = torch.FloatTensor(np.array(y_batch)).transpose(2, 1)  # (B, 1, T)
        c_batch = torch.FloatTensor(np.array(c_batch)).transpose(2, 1)  # (B, C, T')
        d_batch = torch.FloatTensor(np.array(d_batch)).transpose(2, 1)  # (B, 1, T)

        # make input noise signal batch tensor
        if self.input_type == "noise":
            z_batch = torch.randn(y_batch.size())  # (B, 1, T)
            return (z_batch, c_batch, d_batch), y_batch
        else:
            raise NotImplementedError("Currently only 'noise' input is supported ")

    @staticmethod
    def _check_length(x, c, d, hop_size, context_window):
        """Assert the audio and feature lengths are correctly adjusted for upsamping."""
        assert len(x) == (len(c) - 2 * context_window) * hop_size
        assert len(x) == len(d)


def main():
    """Run training process."""
    parser = argparse.ArgumentParser(
        description="Train Quasi-Periodic Parallel WaveGAN (See detail in qppwg/bin/train.py).")
    parser.add_argument("--train_audio", required=True, type=str,
                        help="list of training wav files")
    parser.add_argument("--train_feat", required=True, type=str,
                        help="list of training feat files")
    parser.add_argument("--valid_audio", required=True, type=str,
                        help="list of validation wav files")
    parser.add_argument("--valid_feat", required=True, type=str,
                        help="list of validation feat files")
    parser.add_argument("--stats", required=True, type=str, 
                        help="hdf5 file including statistics")
    parser.add_argument("--outdir", required=True, type=str,
                        help="directory to save checkpoints.")
    parser.add_argument("--config", required=True, type=str,
                        help="yaml format configuration file.")
    parser.add_argument("--pretrain", default="", type=str, nargs="?",
                        help="checkpoint file path to load pretrained params. (default=\"\")")
    parser.add_argument("--resume", default="", type=str, nargs="?",
                        help="checkpoint file path to resume training. (default=\"\")")
    parser.add_argument("--verbose", default=1, type=int,
                        help="logging level. higher is more logging. (default=1)")
    parser.add_argument("--rank", "--local_rank", default=0, type=int,
                        help="rank for distributed training. no need to explictly specify.")
    parser.add_argument("--seed", default=1, type=int, 
                        help="seed number")
    args = parser.parse_args()

    args.distributed = False
    if not torch.cuda.is_available():
        print("CPU")
        device = torch.device("cpu")
    else:
        print("GPU")
        device = torch.device("cuda")
        # effective when using fixed size inputs
        # see https://discuss.pytorch.org/t/what-does-torch-backends-cudnn-benchmark-do/5936
        torch.backends.cudnn.benchmark = True
        torch.cuda.set_device(args.rank)
        # setup for distributed training
        # see example: https://github.com/NVIDIA/apex/tree/master/examples/simple/distributed
        if "WORLD_SIZE" in os.environ:
            args.world_size = int(os.environ["WORLD_SIZE"])
            args.distributed = args.world_size > 1
        if args.distributed:
            torch.distributed.init_process_group(backend="nccl", init_method="env://")

    # fix seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)

    # suppress logging for distributed training
    if args.rank != 0:
        sys.stdout = open(os.devnull, "w")

    # set logger
    if args.verbose > 1:
        logging.basicConfig(
            level=logging.DEBUG, stream=sys.stdout,
            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s")
    elif args.verbose > 0:
        logging.basicConfig(
            level=logging.INFO, stream=sys.stdout,
            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s")
    else:
        logging.basicConfig(
            level=logging.WARN, stream=sys.stdout,
            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s")
        logging.warning("Skip DEBUG/INFO messages")

    # check directory existence
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

    # load and save config
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.Loader)
    config.update(vars(args))
    config["version"] = qppwg.__version__  # add version info
    with open(os.path.join(args.outdir, "config.yml"), "w") as f:
        yaml.dump(config, f, Dumper=yaml.Dumper)
    for key, value in config.items():
        logging.info(f"{key} = {value}")

    # get dataset
    if config["remove_short_samples"]:
        feat_length_threshold = config["batch_max_steps"] // config["hop_size"] + \
            2 * config["generator_params"].get("aux_context_window", 0)
    else:
        feat_length_threshold = None

    audio_load_fn = sf.read
    feat_load_fn = lambda x: read_hdf5(x, config.get("feat_type", "world"))

    train_dataset = AudioFeatDataset(
        stats=args.stats,
        audio_list=args.train_audio,
        audio_load_fn=audio_load_fn,
        feat_list=args.train_feat,
        feat_load_fn=feat_load_fn,
        feat_length_threshold=feat_length_threshold,
        allow_cache=config.get("allow_cache", False),
        hop_size=config["hop_size"],
        dense_factor=config.get("dense_factor", 4),
        f0_threshold=config.get("f0_threshold", 0),
        f0_cont=config.get("f0_cont", True),
        f0_dim_idx=config.get("f0_dim_idx", 1),
        uv_dim_idx=config.get("uv_dim_idx", 0),
        mean_path=config.get("mean_path", "/world/mean"),
        scale_path=config.get("scale_path", "/world/scale"),
        shift=config.get("stats_shift", 1),
    )
    logging.info(f"The number of training files = {len(train_dataset)}.")

    valid_dataset = AudioFeatDataset(
        stats=args.stats,
        audio_list=args.valid_audio,
        audio_load_fn=audio_load_fn,
        feat_list=args.valid_feat,
        feat_load_fn=feat_load_fn,
        feat_length_threshold=feat_length_threshold,
        allow_cache=config.get("allow_cache", False),
        hop_size=config["hop_size"],
        dense_factor=config.get("dense_factor", 4),
        f0_threshold=config.get("f0_threshold", 0),
        f0_cont=config.get("f0_cont", True),
        f0_dim_idx=config.get("f0_dim_idx", 0),
        uv_dim_idx=config.get("uv_dim_idx", 1),
        mean_path=config.get("mean_path", "/world/mean"),
        scale_path=config.get("scale_path", "/world/scale"),
        shift=config.get("stats_shift", 1),
    )
    logging.info(f"The number of validation files = {len(valid_dataset)}.")

    dataset = {
        "train": train_dataset,
        "valid": valid_dataset,
    }

    # get data loader
    collater = Collater(
        batch_max_steps=config["batch_max_steps"],
        hop_size=config["hop_size"],
        # keep compatibility
        aux_context_window=config["generator_params"].get("aux_context_window", 0),
        # keep compatibility
        input_type=config.get("input_type", "noise"),
    )
    train_sampler, valid_sampler = None, None
    if args.distributed:
        # setup sampler for distributed training
        from torch.utils.data.distributed import DistributedSampler
        train_sampler = DistributedSampler(
            dataset=dataset["train"],
            num_replicas=args.world_size,
            rank=args.rank,
            shuffle=True,
        )
        valid_sampler = DistributedSampler(
            dataset=dataset["valid"],
            num_replicas=args.world_size,
            rank=args.rank,
            shuffle=False,
        )

    data_loader = {
        "train": DataLoader(
            dataset=dataset["train"],
            shuffle=False if args.distributed else True,
            collate_fn=collater,
            batch_size=config["batch_size"],
            num_workers=config["num_workers"],
            sampler=train_sampler,
            pin_memory=config["pin_memory"],
        ),
        "valid": DataLoader(
            dataset=dataset["valid"],
            shuffle=False if args.distributed else True,
            collate_fn=collater,
            batch_size=config["batch_size"],
            num_workers=config["num_workers"],
            sampler=valid_sampler,
            pin_memory=config["pin_memory"],
        ),
    }

    # define models and optimizers
    generator_class = getattr(
        qppwg.models,
        # keep compatibility
        config.get("generator_type", "QPPWGGenerator"),
    )
    discriminator_class = getattr(
        qppwg.models,
        # keep compatibility
        config.get("discriminator_type", "QPPWGDiscriminator"),
    )
    model = {
        "generator": generator_class(
            **config["generator_params"]).to(device),
        "discriminator": discriminator_class(
            **config["discriminator_params"]).to(device),
    }
    criterion = {
        "stft": MultiResolutionSTFTLoss(
            **config["stft_loss_params"]).to(device),
        "mse": torch.nn.MSELoss().to(device),
    }
    if config.get("use_feat_match_loss", False):  # keep compatibility
        criterion["l1"] = torch.nn.L1Loss().to(device)
    generator_optimizer_class = getattr(
        qppwg.optimizers,
        # keep compatibility
        config.get("generator_optimizer_type", "RAdam"),
    )
    discriminator_optimizer_class = getattr(
        qppwg.optimizers,
        # keep compatibility
        config.get("discriminator_optimizer_type", "RAdam"),
    )
    optimizer = {
        "generator": generator_optimizer_class(
            model["generator"].parameters(),
            **config["generator_optimizer_params"],
        ),
        "discriminator": discriminator_optimizer_class(
            model["discriminator"].parameters(),
            **config["discriminator_optimizer_params"],
        ),
    }
    generator_scheduler_class = getattr(
        torch.optim.lr_scheduler,
        # keep compatibility
        config.get("generator_scheduler_type", "StepLR"),
    )
    discriminator_scheduler_class = getattr(
        torch.optim.lr_scheduler,
        # keep compatibility
        config.get("discriminator_scheduler_type", "StepLR"),
    )
    scheduler = {
        "generator": generator_scheduler_class(
            optimizer=optimizer["generator"],
            **config["generator_scheduler_params"],
        ),
        "discriminator": discriminator_scheduler_class(
            optimizer=optimizer["discriminator"],
            **config["discriminator_scheduler_params"],
        ),
    }
    if args.distributed:
        # wrap model for distributed training
        try:
            from apex.parallel import DistributedDataParallel
        except ImportError:
            raise ImportError("apex is not installed. please check https://github.com/NVIDIA/apex.")
        model["generator"] = DistributedDataParallel(model["generator"])
        model["discriminator"] = DistributedDataParallel(model["discriminator"])
    logging.debug(model["generator"])
    logging.debug(model["discriminator"])

    # define trainer
    trainer = Trainer(
        steps=0,
        epochs=0,
        data_loader=data_loader,
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        config=config,
        device=device,
    )

    # load pretrained/resume parameters from checkpoint
    if os.path.exists(args.resume):
        trainer.load_checkpoint(args.resume)
        logging.info(f"Successfully resumed from {args.resume}.")
    elif os.path.exists(args.pretrain):
        trainer.load_checkpoint(args.pretrain, load_only_params=True)
        logging.info(f"Successfully load parameters from {args.pretrain}.")
    else:
        logging.info("Start a new training process.")
        
    # run training loop
    try:
        trainer.run()
    except KeyboardInterrupt:
        trainer.save_checkpoint(
            os.path.join(config["outdir"], f"checkpoint-{trainer.steps}steps.pkl"))
        logging.info(f"Successfully saved checkpoint @ {trainer.steps}steps.")


if __name__ == "__main__":
    main()
