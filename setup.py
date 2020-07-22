#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Setup QPPWG library."""

import os
import pip
import sys

from distutils.version import LooseVersion
from setuptools import find_packages
from setuptools import setup

if LooseVersion(sys.version) < LooseVersion("3.6"):
    raise RuntimeError(
        "qppwg requires Python>=3.6, "
        "but your Python is {}".format(sys.version))
if LooseVersion(pip.__version__) < LooseVersion("19"):
    raise RuntimeError(
        "pip>=19.0.0 is required, but your pip is {}. "
        "Try again after \"pip install -U pip\"".format(pip.__version__))

requirements = {
    "install": [
        "torch>=1.0.1",
        "setuptools>=38.5.1",
        "librosa>=0.8.0",
        "soundfile>=0.10.2",
        "tensorboardX>=1.8",
        "matplotlib>=3.1.0",
        "PyYAML>=3.12",
        "tqdm>=4.26.1",
        "kaldiio>=2.14.1",
        "h5py>=2.10.0",
        "docopt",
        "sprocket-vc",
    ],
    "setup": [
        "numpy",
        "pytest-runner",
    ],
    "test": [
        "pytest>=3.3.0",
        "hacking>=1.1.0",
        "flake8>=3.7.8",
        "flake8-docstrings>=1.3.1",
    ]
}
entry_points = {
    "console_scripts": [
        "qppwg-preprocess=qppwg.bin.preprocess:main",
        "qppwg-compute-statistics=qppwg.bin.compute_statistics:main",
        "qppwg-train=qppwg.bin.train:main",
        "qppwg-decode=qppwg.bin.decode:main",
    ]
}

install_requires = requirements["install"]
setup_requires = requirements["setup"]
tests_require = requirements["test"]
extras_require = {k: v for k, v in requirements.items()
                  if k not in ["install", "setup"]}

dirname = os.path.dirname(__file__)
setup(name="qppwg",
      version="0.1.2",
      url="http://github.com/bigpon/QPPWG",
      author="Yi-Chiao Wu",
      author_email="yichiao.wu@g.sp.m.is.nagoya-u.ac.jp",
      description="Quasi-Periodic Parallel WaveGAN implementation",
      long_description_content_type="text/markdown",
      long_description=open(os.path.join(dirname, "README.md"), encoding="utf-8").read(),
      license="MIT License",
      packages=find_packages(include=["qppwg*"]),
      install_requires=install_requires,
      setup_requires=setup_requires,
      tests_require=tests_require,
      extras_require=extras_require,
      entry_points=entry_points,
      classifiers=[
          "Programming Language :: Python :: 3.6",
          "Programming Language :: Python :: 3.7",
          "Intended Audience :: Science/Research",
          "Operating System :: POSIX :: Linux",
          "License :: OSI Approved :: MIT License",
          "Topic :: Software Development :: Libraries :: Python Modules"],
      )
