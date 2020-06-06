# -*- coding: utf-8 -*-

# Copyright 2020 Yi-Chiao Wu (Nagoya University)
#  MIT License (https://opensource.org/licenses/MIT)

"""Feature-related functions."""

import numpy as np


def validate_length(x, y, hop_size=None):
    """ Validate length
    Args:
        x (ndarray): numpy array with x.shape[0] = len_x
        y (ndarray): numpy array with y.shape[0] = len_y
        hop_size (int): upsampling factor
    Returns:
        (ndarray): length adjusted x with same length y
        (ndarray): length adjusted y with same length x
    """
    if hop_size is None:
        if x.shape[0] < y.shape[0]:
            y = y[:x.shape[0]]
        if x.shape[0] > y.shape[0]:
            x = x[:y.shape[0]]
        assert len(x) == len(y)
    else:
        if x.shape[0] > y.shape[0] * hop_size:
            x = x[:y.shape[0] * hop_size]
        if x.shape[0] < y.shape[0] * hop_size:
            mod_y = y.shape[0] * hop_size - x.shape[0]
            mod_y_frame = mod_y // hop_size + 1
            y = y[:-mod_y_frame]
            x = x[:y.shape[0] * hop_size]
        assert len(x) == len(y) * hop_size

    return x, y


def batch_f0(h, f0_threshold=0, f0_cont=True, f0_idx=1, uv_idx=0):
    """ load f0 
    Args:
        h (ndarray): the auxiliary acoustic features (T x D)
        f0_threshold (float): the lower bound of pitch
        f0_cont (bool): True: return continuous f0; False return discrete f0
        f0_idx: the dimension index of f0
        uv_idx: the dimension index of U/V
    Return:
        f0(ndarray): 
            float array of the f0 sequence (T)
    """
    if (f0_idx < 0) or (uv_idx < 0):
        f0 = np.zeros(h.shape[0])
    else:
        f0 = h[:, f0_idx].copy(order='C')
        f0[f0 < f0_threshold] = f0_threshold
        if not f0_cont:
            uv = h[:, uv_idx].copy(order='C')  # voice/unvoice feature
            f0[uv == 0] = 0

    return f0


def dilated_factor(batch_f0, fs, dense_factor):
    """Pitch-dependent dilated factor
    Args:
        batch_f0 (ndarray): the f0 sequence (T)
        fs (int): sampling rate
        dense_factor (int): the number of taps in one cycle
    Return:
        dilated_factors(np array): 
            float array of the pitch-dependent dilated factors (T)
    """
    batch_f0[batch_f0 == 0] = fs / dense_factor
    dilated_factors = np.ones(batch_f0.shape) * fs
    dilated_factors /= batch_f0
    dilated_factors /= dense_factor
    assert np.all(dilated_factors > 0)

    return dilated_factors
