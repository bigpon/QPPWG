#! /bin/bash
# -*- coding: utf-8 -*-

# Copyright 2020 Yi-Chiao Wu (Nagoya University)
#  MIT License (https://opensource.org/licenses/MIT)

trainset=vcc20_train_24kHz  # training set
validset=vcc20_valid_24kHz  # validation set
evalset=vcc20_eval_24kHz    # evaluation set
gpu=0           # gpu id
conf=PWG_30     # name of config
resume=None     # number of iteration of resume model
iter=400000     # number of iteration of testing model
scaled=0.50     # scaled ratio of f0
stage=          # running stage (0-3)
                # stage 0: Preprocessing
                # stage 1: QPPWG training
                # stage 2: QPPWG decoding (analysis-synthesis)
                # stage 3: QPPWG decoding (scaled F0)
. ../parse_options.sh || exit 1;

# Preprocessing
if echo ${stage} | grep -q 0; then
    echo "Preprocessing."
    python run.py -C ${conf} -T ${trainset} -V ${validset} -E ${evalset} -0
fi

# QPPWG training
if echo ${stage} | grep -q 1; then
    echo "QPPWG training."
    python run.py -g ${gpu} -C ${conf} \
    -T ${trainset} -V ${validset} -R ${resume} -1
fi

# QPPWG decoding w/ natural acoustic features
if echo ${stage} | grep -q 2; then
    echo "QPPWG decoding (natural)."
    python run.py -g ${gpu} -C ${conf} \
    -T ${trainset} -E ${evalset} -I ${iter} -2
fi

# QPPWG decoding w/ scaled F0
if echo ${stage} | grep -q 3; then
    echo "QPPWG decoding ( ${scaled} x F0)."
    python run.py -g ${gpu} -C ${conf} -f ${scaled}\
    -T ${trainset} -E ${evalset} -I ${iter} -2
fi