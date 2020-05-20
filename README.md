
# Quasi-Periodic Parallel WaveGAN (QPPWG)

[![](https://img.shields.io/pypi/v/qppwg)](https://pypi.org/project/qppwg/) ![](https://img.shields.io/pypi/pyversions/qppwg) ![](https://img.shields.io/pypi/l/qppwg)

This is official [QPPWG](https://arxiv.org/abs/2005.08654) PyTorch implementation.
QPPWG is a non-autoregressive neural speech generation model developed based on [PWG](https://ieeexplore.ieee.org/abstract/document/9053795) and a [QP](https://bigpon.github.io/QuasiPeriodicWaveNet_demo) structure.

![](https://user-images.githubusercontent.com/10822486/82352944-af1dca80-9a39-11ea-806d-1aa6a91d2773.png)

In this repo, we provide an example to train and test QPPWG as a vocoder for [WORLD](https://doi.org/10.1587/transinf.2015EDP7457) acoustic features.
More details can be found on our [Demo](https://bigpon.github.io/QuasiPeriodicParallelWaveGAN_demo) page.


## News
- **2020/5/20** release the first version.


## Requirements

This repository is tested on Ubuntu 16.04 with a GPU Titan V.

- Python 3.6+
- Cuda 10.0
- CuDNN 7+
- PyTorch 1.0.1+


## Environment setup

The code works with both anaconda and virtualenv.
The following example uses anaconda.

```bash
$ conda create -n venvQPPWG python=3.6
$ source activate venvQPPWG
$ git clone https://github.com/bigpon/QPPWG.git
$ cd QPPWG
$ pip install -e .
```

More details can refer to the [PWG](https://github.com/kan-bayashi/ParallelWaveGAN) repo.


## Folder architecture
- **egs**
the folder for projects.
- **egs/vcc18**
the folder of the VCC2018 project.
- **egs/vcc18/exp**
the folder for trained models.
- **egs/vcc18/conf**
the folder for configs.
- **egs/vcc18/data**
the folder for corpus related files (wav, feature, list ...).
- **qppwg**
the folder of the source codes.


## Run

### Corpus and path setup

- Modify the corresponding CUDA paths in `egs/vcc18/run.py`.
- Download the [Voice Conversion Challenge 2018](https://datashare.is.ed.ac.uk/handle/10283/3061) (VCC2018) corpus to run the QPPWG example

```bash
$ cd egs/vcc18
# Download training and validation corpus
$ wget -o train.log -O data/wav/train.zip https://datashare.is.ed.ac.uk/bitstream/handle/10283/3061/vcc2018_database_training.zip
# Download evaluation corpus
$ wget -o eval.log -O data/wav/eval.zip https://datashare.is.ed.ac.uk/bitstream/handle/10283/3061/vcc2018_database_evaluation.zip
# unzip corpus
$ unzip data/wav/train.zip -d data/wav/
$ unzip data/wav/eval.zip -d data/wav/
```

- **Training wav lists**: `data/scp/vcc18_train_22kHz.scp`.
- **Validation wav lists**: `data/scp/vcc18_valid_22kHz.scp`.
- **Testing wav list**: `data/scp/vcc18_eval_22kHz.scp`.

### Preprocessing

```bash
# Extract WORLD acoustic features and statistics of training and testing data
$ bash run.sh --stage 0 --config PWG_30
```

- WORLD-related settings can be changed in `egs/vcc18/conf/vcc18.PWG_30.yaml`.
- If you want to extract other corpus, please create a corresponding config and a file including power thresholds and f0 ranges like `egs/vcc18/data/pow_f0_dict.yml`.
- More details about feature extraction can refer to the [QPNet](https://github.com/bigpon/QPNet) repo.
- The lists of auxiliary features will be automatically generated.
- **Training aux lists**: `data/scp/vcc18_train_22kHz.list`.
- **Validation aux lists**: `data/scp/vcc18_valid_22kHz.list`.
- **Testing aux list**: `data/scp/vcc18_eval_22kHz.list`.


### QPPWG/PWG training

```bash
# Training a QPPWG model with the 'QPPWGaf_20' config and the 'vcc18_train_22kHz' and 'vcc18_valid_22kHz' sets.
$ bash run.sh --gpu 0 --stage 1 --conf QPPWGaf_20 \
--trainset vcc18_train_22kHz --validset vcc18_valid_22kHz
```

- The gpu ID can be set by --gpu GPU_ID (default: 0)
- The model architecture can be set by --conf CONFIG (default: PWG_30)
- The trained model resume can be set by --resume NUM (default: None)


### QPPWG/PWG testing

```bash
# QPPWG/PWG decoding w/ natural acoustic features
$ bash run.sh --gpu 0 --stage 2 --conf QPPWGaf_20 \
--iter 400000 --trainset vcc18_train_22kHz --evalset vcc18_eval_22kHz
# QPPWG/PWG decoding w/ scaled f0 (ex: halved f0).
$ bash run.sh --gpu 0 --stage 3 --conf QPPWGaf_20 --scaled 0.50 \
--iter 400000 --trainset vcc18_train_22kHz --evalset vcc18_eval_22kHz
```

### Monitor training progress

```bash
$ tensorboard --logdir exp
```

- The RTF of PWG_30 decoding with a TITAN V is **0.016**.
- The RTF of PWG_20 decoding with a TITAN V is **0.011**.
- The RTF of QPPWGaf_20 decoding with a TITAN V is **0.018**.
- The training time of PWG_30 with a TITAN V is around 3 days.
- The training time of QPPWGaf_20 with a TITAN V is around 5 days.


## Results
[TODO] We will release the pre-trained models and all generated samples around June 2020.


## References
The QPPWG repository is developed based on the following repositories and paper.

- [kan-bayashi/ParallelWaveGAN](https://github.com/kan-bayashi/ParallelWaveGAN)
- [bigpon/QPNet](https://github.com/bigpon/QPNet)
- [k2kobayashi/sprocket](https://github.com/k2kobayashi/sprocket)
- [r9y9/wavenet_vocoder](https://github.com/r9y9/wavenet_vocoder)
- [Parallel WaveGAN](https://arxiv.org/abs/1910.11480)



## Citation

If you find the code is helpful, please cite the following article.

```
@article{wu2020qppwg,
title={Quasi-Periodic Parallel WaveGAN Vocoder: A Non-autoregressive Pitch-dependent   Dilated Convolution Model for Parametric Speech Generation},
author={Wu, Yi-Chiao and Hayashi, Tomoki and Okamoto, Takuma and Kawai, Hisashi and Toda, Tomoki},
journal={arXiv preprint arXiv:2005.08654},
year={2020}
}
```


## Authors

Development:
Yi-Chiao Wu @ Nagoya University ([@bigpon](https://github.com/bigpon))
E-mail: `yichiao.wu@g.sp.m.is.nagoya-u.ac.jp`

Advisor:
Tomoki Toda @ Nagoya University
E-mail: `tomoki@icts.nagoya-u.ac.jp`