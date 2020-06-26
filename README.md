
# Quasi-Periodic Parallel WaveGAN (QPPWG)

[![](https://img.shields.io/pypi/v/qppwg)](https://pypi.org/project/qppwg/) ![](https://img.shields.io/pypi/pyversions/qppwg) ![](https://img.shields.io/pypi/l/qppwg)


This is official [QPPWG](https://arxiv.org/abs/2005.08654) PyTorch implementation.
QPPWG is a non-autoregressive neural speech generation model developed based on [PWG](https://ieeexplore.ieee.org/abstract/document/9053795) and a [QP](https://bigpon.github.io/QuasiPeriodicWaveNet_demo) structure.

<p align="center">
<img src="https://user-images.githubusercontent.com/10822486/82352944-af1dca80-9a39-11ea-806d-1aa6a91d2773.png"/>
</p>

In this repo, we provide an example to train and test QPPWG as a vocoder for [WORLD](https://doi.org/10.1587/transinf.2015EDP7457) acoustic features.
More details can be found on our [Demo](https://bigpon.github.io/QuasiPeriodicParallelWaveGAN_demo) page.


## News
<!--- **2020/6/30** Release the pre-trained models of [vcc20](http://www.vc-challenge.org/) corpus.-->
- **2020/6/26** Release the pre-trained models of [vcc18](http://www.vc-challenge.org/vcc2018/index.html) corpus.
- **2020/5/20** Release the first version.


## Requirements

This repository is tested on Ubuntu 16.04 with a Titan V GPU.

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

Please refer to the [PWG](https://github.com/kan-bayashi/ParallelWaveGAN) repo for more details.


## Folder architecture
- **egs**:  
The folder for projects.
- **egs/vcc18**:  
The folder of the VCC2018 project.
- **egs/vcc18/exp**:  
The folder for trained models.
- **egs/vcc18/conf**:  
The folder for configs.
- **egs/vcc18/data**:  
The folder for corpus related files (wav, feature, list ...).
- **qppwg**:  
The folder of the source codes.


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
- If you want to use another corpus, please create a corresponding config and a file including power thresholds and f0 ranges like `egs/vcc18/data/pow_f0_dict.yml`.
- More details about feature extraction can be found in the [QPNet](https://github.com/bigpon/QPNet) repo.
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

- The training time of PWG_30 with a TITAN V is around 3 days.
- The training time of QPPWGaf_20 with a TITAN V is around 5 days.


## Inference speed (RTF)

- Vanilla PWG (PWG_30)

```bash
# On CPU (Intel(R) Xeon(R) Gold 6142 CPU @ 2.60GHz 32 threads)
[decode]: 100%|███████████| 140/140 [04:50<00:00,  2.08s/it, RTF=0.771]
2020-05-26 12:30:27,273 (decode:156) INFO: Finished generation of 140 utterances (RTF = 0.579).
# On GPU (TITAN V)
[decode]: 100%|███████████| 140/140 [00:09<00:00, 14.89it/s, RTF=0.0155]
2020-05-26 12:32:26,160 (decode:156) INFO: Finished generation of 140 utterances (RTF = 0.016).
```

- PWG w/ only 20 blocks (PWG_20)

```bash
# On CPU (Intel(R) Xeon(R) Gold 6142 CPU @ 2.60GHz 32 threads)
[decode]: 100%|███████████| 140/140 [03:57<00:00,  1.70s/it, RTF=0.761]
2020-05-30 13:50:20,438 (decode:156) INFO: Finished generation of 140 utterances (RTF = 0.474).
# On GPU (TITAN V)
[decode]: 100%|███████████| 140/140 [00:08<00:00, 16.55it/s, RTF=0.0105]
2020-05-30 13:43:50,793 (decode:156) INFO: Finished generation of 140 utterances (RTF = 0.011).
```

- QPPWG (QPPWGaf_20)  

```bash
# On CPU (Intel(R) Xeon(R) Gold 6142 CPU @ 2.60GHz 32 threads)
[decode]: 100%|███████████| 140/140 [04:12<00:00,  1.81s/it, RTF=0.455]
2020-05-26 12:38:15,982 (decode:156) INFO: Finished generation of 140 utterances (RTF = 0.512).
# On GPU (TITAN V)
[decode]: 100%|███████████| 140/140 [00:11<00:00, 12.57it/s, RTF=0.0218]
2020-05-26 12:33:32,469 (decode:156) INFO: Finished generation of 140 utterances (RTF = 0.020).
```


## Models and results

[TODO] We will release mel-spectrogram extraction and vcc20 pre-trained model.

- The pre-trained models and generated utterances are released.  
- You can download the whole folder of each corpus and then put it in `egs/[corpus]` to run speech generations with the pre-trained models.  
- You also can only download the `[corpus]/data` folder and the desired pre-trained model and then put the `data` folder in `egs/[corpus]` and the model folder in `egs/[corpus]/exp`.  
- Both models with 100,000 iterations (trained w/ only STFT loss) and 400,000 iterations (trained w/ STFT and GAN losses) are released.  
- The generated utterances are in the `wav` folder of each model’s folder. (Only the vcc18 results are released now.)

<p align="center">
<table class="tg">
<thead>
  <tr>
    <th class="tg-0lax">Corpus</th>
    <th class="tg-0lax">Lang</th>
    <th class="tg-0lax">Fs [Hz]</th>
    <th class="tg-0lax">Feature</th>
    <th class="tg-0lax">Model</th>
    <th class="tg-0lax">Conf</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-0pky" rowspan="3">
    <a href="https://drive.google.com/drive/folders/1WFqk08lJE4LrYocUxZo7cdT7_BxGitiL?usp=sharing">
    vcc18</a></td>
    <td class="tg-0pky" rowspan="3">EN</td>
    <td class="tg-0pky" rowspan="3">22050</td>
    <td class="tg-0pky" rowspan="3">world</td>
    <td class="tg-0pky">
    <a href="https://drive.google.com/drive/folders/1kTQ0iYBy7t7EnxFiwhh7gII1KVDpuvts?usp=sharing">
    PWG_20</a></td>
    <td class="tg-0pky">
    <a href="https://github.com/bigpon/QPPWG/blob/master/egs/vcc18/conf/vcc18.PWG_20.yaml">
    link</a></td>
  </tr>
  <tr>
    <td class="tg-0pky">
    <a href="https://drive.google.com/drive/folders/1HHCgFpqJQO9NnrDkZdNKvw4WeCm_xPWl?usp=sharing">
    PWG_30</td>
    <td class="tg-0pky">
    <a href="https://github.com/bigpon/QPPWG/blob/master/egs/vcc18/conf/vcc18.PWG_30.yaml">
    link</td>
  </tr>
  <tr>
    <td class="tg-0pky">
    <a href="https://drive.google.com/drive/folders/12kbJNKjqJwcImc4iTcu6J53s4st297bD?usp=sharing">
    QPPWGaf_20</td>
    <td class="tg-0pky">
    <a href="https://github.com/bigpon/QPPWG/blob/master/egs/vcc18/conf/vcc18.QPPWGaf_20.yaml">
    link</td>
  </tr>
  <!--
  <tr>
    <td class="tg-0pky" rowspan="3">
    <a href="https://drive.google.com/drive/folders/1khnMmwY-_6HzNtZgmT2xwoWgC6MYuKLZ?usp=sharing">
    vcc20</td>
    <td class="tg-0pky" rowspan="3">EN</td>
    <td class="tg-0pky" rowspan="3">24000</td>
    <td class="tg-0pky" rowspan="3">melfb + f0</td>
    <td class="tg-0pky">
    <a href="https://drive.google.com/drive/folders/1TTJMGyzHSSLzqqFcQwYyjsKwcW48S9zc?usp=sharing">
    PWG_20</td>
    <td class="tg-0pky">
    <a href="https://github.com/bigpon/QPPWG/blob/master/egs/vcc20/conf/vcc20.PWG_20.yaml">
    link</td>
  </tr>
  <tr>
    <td class="tg-0pky">
    <a href="https://drive.google.com/drive/folders/1rrHVtBQRqclsskBJi6IErF-FqOqb6v37?usp=sharing">
    PWG_30</td>
    <td class="tg-0pky">
    <a href="https://github.com/bigpon/QPPWG/blob/master/egs/vcc20/conf/vcc20.PWG_30.yaml">
    link</td>
  </tr>
  <tr>
    <td class="tg-0pky">
    <a href="https://drive.google.com/drive/folders/1UXZG17xsE3MCqroAe_5vj49aeW7xEH6M?usp=sharing">
    QPPWGaf_20</td>
    <td class="tg-0pky">
    <a href="https://github.com/bigpon/QPPWG/blob/master/egs/vcc20/conf/vcc20.QPPWGaf_20.yaml">
    link</td>
  </tr>
  -->
</tbody>
</table> 
</p>


## Usage of pre-trained models

### Analysis-synthesis

The minimum code for performing analysis and synthesis is presented.

```bash
# Make sure you have installed `qppwg`
# If not, install it via pip
$ pip install qppwg
# Take "vcc18" corpus as an example
# Download the whole folder of "vcc18"
$ ls vcc18
  data    exp
# Change directory to `vcc18` folder
$ cd vcc18
# Put audio files in `data/wav/` directory
$ ls data/wav/
  sample1.wav    sample2.wav
# Create a list `data/sample.scp` of the audio files
$ tail data/scp/sample.scp
  data/wav/sample1.wav
  data/wav/sample2.wav
# Extract acoustic features
$ qppwg-preprocess \
    --audio data/scp/sample.scp \
    --indir wav \
    --outdir hdf5 \
    --config exp/qppwg_vcc18_train_22kHz_QPPWGaf_20/config.yml
# The extracted features are in `data/hdf5/`
# The feature list `data/sample.list` of the feature files will be automatically generated
$ ls data/hdf5/
  sample1.h5    sample2.h5
$ ls data/scp/
  sample.scp    sample.list
# Synthesis
$ qppwg-decode \
    --eval_feat data/scp/sample.list \
    --stats data/stats/vcc18_train_22kHz.joblib \
    --indir data/hdf5/ \
    --outdir exp/qppwg_vcc18_train_22kHz_QPPWGaf_20/wav/400000/ \
    --checkpoint exp/qppwg_vcc18_train_22kHz_QPPWGaf_20/checkpoint-400000steps.pkl 
# Synthesis w/ halved F0
$ qppwg-decode \
    --f0_factor 0.50 \
    --eval_feat data/scp/sample.list \
    --stats data/stats/vcc18_train_22kHz.joblib \
    --indir data/hdf5/ \
    --outdir exp/qppwg_vcc18_train_22kHz_QPPWGaf_20/wav/400000/ \
    --checkpoint exp/qppwg_vcc18_train_22kHz_QPPWGaf_20/checkpoint-400000steps.pkl 
# The generated utterances can be found in `exp/[model]/wav/400000/`
$ ls exp/qppwg_vcc18_train_22kHz_QPPWGaf_20/wav/400000/
  sample1.wav    sample1_f0.50.wav    sample2.wav    sample2_f0.50.wav   
```


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
