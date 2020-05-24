# -*- coding: utf-8 -*-

# Copyright 2020 Yi-Chiao Wu (Nagoya University)
# based on a Parallel WaveGAN script by Tomoki Hayashi (Nagoya University)
# (https://github.com/kan-bayashi/ParallelWaveGAN)
#  MIT License (https://opensource.org/licenses/MIT)

"""Parallel WaveGAN Modules."""

import logging
import math
import sys
import torch

from qppwg.layers import Conv1d
from qppwg.layers import Conv1d1x1
from qppwg.layers import FixedBlock
from qppwg.layers import AdaptiveBlock
from qppwg.layers import upsample
from qppwg.utils import pd_indexing, index_initial


class QPPWGGenerator(torch.nn.Module):
    """Quasi-Periodic Parallel WaveGAN Generator module."""

    def __init__(self,
                 in_channels=1,
                 out_channels=1,
                 blockF=10,
                 cycleF=1,
                 blockA=10,
                 cycleA=2,
                 cascade_mode=1,
                 residual_channels=64,
                 gate_channels=128,
                 skip_channels=64,
                 aux_channels=80,
                 aux_context_window=2,
                 upsample_params={"upsample_scales": [5, 2, 11, 1]},
                 ):
        """Initialize Parallel WaveGAN Generator module.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            blockF (int): Number of fixed residual blocks.
            cycleF (int): Number of dilation cycles of fixed residual blocks.
            blockA (int): Number of adaptive residual blocks.
            cycleA (int): Number of dilation cycles of adaptive residual blocks.
            cascade_mode (int): Cascaded mode (0: Fixed->Adaptive; 1: Adaptive->Fixed).
            residual_channels (int): Number of channels in residual conv.
            gate_channels (int):  Number of channels in gated conv.
            skip_channels (int): Number of channels in skip conv.
            aux_channels (int): Number of channels for auxiliary feature conv.
            aux_context_window (int): Context window size for auxiliary feature.
            upsample_params (dict): Upsampling network parameters.

        """
        super(QPPWGGenerator, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.aux_channels = aux_channels
        self.n_ch = residual_channels

        # check the number of blocks and cycles
        cycleA = max(cycleA, 1)
        cycleF = max(cycleF, 1)
        assert blockF % cycleF == 0
        blockF_per_cycle = blockF // cycleF
        assert blockA % cycleA == 0
        self.blockA_per_cycle = blockA // cycleA

        # define first convolution
        self.conv_first = Conv1d1x1(in_channels, residual_channels, bias=True)

        # define upsampling network
        upsample_params.update({
            "aux_channels": aux_channels,
            "aux_context_window": aux_context_window,
        })
        self.upsample_net = getattr(upsample, "ConvInUpsampleNetwork")(**upsample_params)
        
        # define fixed residual blocks
        fixed_blocks = torch.nn.ModuleList()
        for block in range(blockF):
            dilation = 2 ** (block % blockF_per_cycle)
            conv = FixedBlock(
                residual_channels=residual_channels,
                gate_channels=gate_channels,
                skip_channels=skip_channels,
                aux_channels=aux_channels,
                dilation=dilation,
                bias=True,
            )
            fixed_blocks += [conv]
        
        # define adaptive residual blocks
        adaptive_blocks = torch.nn.ModuleList()
        for block in range(blockA):
            conv = AdaptiveBlock(
                residual_channels=residual_channels,
                gate_channels=gate_channels,
                skip_channels=skip_channels,
                aux_channels=aux_channels,
                bias=True,
            )
            adaptive_blocks += [conv]
        
        # define cascaded structure
        if cascade_mode == 0:  # fixed->adaptive
            self.conv_dilated = fixed_blocks.extend(adaptive_blocks)
            self.block_modes = [False] * blockF + [True] * blockA
        elif cascade_mode == 1:  # adaptive->fixed
            self.conv_dilated = adaptive_blocks.extend(fixed_blocks)
            self.block_modes = [True] * blockA + [False] * blockF
        else:
            logging.error("Cascaded mode %d is not supported!" % (cascade_mode))
            sys.exit(0)

        # define output layers
        self.conv_last = torch.nn.ModuleList([
            torch.nn.ReLU(inplace=True),
            Conv1d1x1(skip_channels, skip_channels, bias=True),
            torch.nn.ReLU(inplace=True),
            Conv1d1x1(skip_channels, out_channels, bias=True),
        ])

        # apply weight norm
        self.apply_weight_norm()

    def forward(self, x, c, d):
        """Calculate forward propagation.

        Args:
            x (Tensor): Input noise signal (B, 1, T).
            c (Tensor): Local conditioning auxiliary features (B, C ,T').
            d (Tensor): Input pitch-dependent dilated factors (B, 1, T).

        Returns:
            Tensor: Output tensor (B, out_channels, T)

        """
        # index initialization
        batch_index, ch_index = index_initial(x.size(0), self.n_ch)

        # perform upsampling
        c = self.upsample_net(c)
        assert c.size(-1) == x.size(-1)

        # encode to hidden representation
        x = self.conv_first(x)
        skips = 0
        blockA_idx = 0
        for f, mode in zip(self.conv_dilated, self.block_modes):
            if mode:  # adaptive block
                dilation = 2 ** (blockA_idx % self.blockA_per_cycle)
                xP, xF = pd_indexing(x, d, dilation, batch_index, ch_index)
                x, h = f(x, xP, xF, c)
                blockA_idx += 1
            else:  # fixed block
                x, h = f(x, c)
            skips += h
        skips *= math.sqrt(1.0 / len(self.conv_dilated))

        # apply final layers
        x = skips
        for f in self.conv_last:
            x = f(x)

        return x

    def remove_weight_norm(self):
        """Remove weight normalization module from all of the layers."""
        def _remove_weight_norm(m):
            try:
                logging.debug(f"Weight norm is removed from {m}.")
                torch.nn.utils.remove_weight_norm(m)
            except ValueError:  # this module didn't have weight norm
                return

        self.apply(_remove_weight_norm)

    def apply_weight_norm(self):
        """Apply weight normalization module from all of the layers."""
        def _apply_weight_norm(m):
            if isinstance(m, torch.nn.Conv1d) or isinstance(m, torch.nn.Conv2d):
                torch.nn.utils.weight_norm(m)
                logging.debug(f"Weight norm is applied to {m}.")

        self.apply(_apply_weight_norm)


class PWGDiscriminator(torch.nn.Module):
    """Parallel WaveGAN Discriminator module."""

    def __init__(self,
                 in_channels=1,
                 out_channels=1,
                 kernel_size=3,
                 layers=10,
                 conv_channels=64,
                 dilation_factor=1,
                 nonlinear_activation="LeakyReLU",
                 nonlinear_activation_params={"negative_slope": 0.2},
                 bias=True,
                 use_weight_norm=True,
                 ):
        """Initialize Parallel WaveGAN Discriminator module.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_size (int): Number of output channels.
            layers (int): Number of conv layers.
            conv_channels (int): Number of chnn layers.
            dilation_factor (int): Dilation factor. For example, if dilation_factor = 2,
                the dilation will be 2, 4, 8, ..., and so on.
            nonlinear_activation (str): Nonlinear function after each conv.
            nonlinear_activation_params (dict): Nonlinear function parameters
            bias (bool): Whether to use bias parameter in conv.
            use_weight_norm (bool) Whether to use weight norm.
                If set to true, it will be applied to all of the conv layers.

        """
        super(PWGDiscriminator, self).__init__()
        assert (kernel_size - 1) % 2 == 0, "Not support even number kernel size."
        assert dilation_factor > 0, "Dilation factor must be > 0."
        self.conv_layers = torch.nn.ModuleList()
        conv_in_channels = in_channels
        for i in range(layers - 1):
            if i == 0:
                dilation = 1
            else:
                dilation = i if dilation_factor == 1 else dilation_factor ** i
                conv_in_channels = conv_channels
            padding = (kernel_size - 1) // 2 * dilation
            conv_layer = [
                Conv1d(conv_in_channels, conv_channels,
                       kernel_size=kernel_size, padding=padding,
                       dilation=dilation, bias=bias),
                getattr(torch.nn, nonlinear_activation)(inplace=True, **nonlinear_activation_params)
            ]
            self.conv_layers += conv_layer
        padding = (kernel_size - 1) // 2
        conv_last_layer = Conv1d(
            conv_in_channels, out_channels,
            kernel_size=kernel_size, padding=padding, bias=bias)
        self.conv_layers += [conv_last_layer]

        # apply weight norm
        if use_weight_norm:
            self.apply_weight_norm()

    def forward(self, x):
        """Calculate forward propagation.

        Args:
            x (Tensor): Input noise signal (B, 1, T).

        Returns:
            Tensor: Output tensor (B, 1, T)

        """
        for f in self.conv_layers:
            x = f(x)
        return x

    def apply_weight_norm(self):
        """Apply weight normalization module from all of the layers."""
        def _apply_weight_norm(m):
            if isinstance(m, torch.nn.Conv1d) or isinstance(m, torch.nn.Conv2d):
                torch.nn.utils.weight_norm(m)
                logging.debug(f"Weight norm is applied to {m}.")

        self.apply(_apply_weight_norm)

    def remove_weight_norm(self):
        """Remove weight normalization module from all of the layers."""
        def _remove_weight_norm(m):
            try:
                logging.debug(f"Weight norm is removed from {m}.")
                torch.nn.utils.remove_weight_norm(m)
            except ValueError:  # this module didn't have weight norm
                return

        self.apply(_remove_weight_norm)
