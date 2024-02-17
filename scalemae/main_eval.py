# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
# flake8: noqa
"""
Example:
    >>> from scalemae.main_eval import *  # NOQA
"""
import argparse
import datetime
import json
import os
import sys
import time
from pathlib import Path
from re import L

import numpy as np
import timm
import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as tv_transforms

import os
import tempfile

import matplotlib.pyplot as plt
import numpy as np
import timm.optim.optim_factory as optim_factory
import yaml

from PIL import Image

from torch.distributed.elastic.multiprocessing.errors import record
from torch.utils.data import DataLoader, Subset

from torchvision import transforms

try:
    from torchgeo.datasets import NAIP, stack_samples
    from torchgeo.datasets.utils import download_url
    from torchgeo.samplers import RandomGeoSampler, Units
except ImportError:
    print('warning: could not import torchgeo')

try:
    import kornia.augmentation as K
    from kornia.augmentation import AugmentationSequential
    from kornia.constants import Resample
except ImportError:
    print('warning: could not import kornia')

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    print('warning: could not import torch.utils.tensorboard')

try:
    import wandb
    from wandb_log import WANDB_LOG_IMG_CONFIG
except ImportError:
    print('warning: could not import kornia')

import scalemae.models_mae
import scalemae.util.misc as misc
from scalemae.util.misc import NativeScalerWithGradNormCount as NativeScaler
from scalemae.util.misc import is_main_process
from scalemae.engine_pretrain import train_one_epoch
from scalemae.lib.scheduler import ConstantResolutionScheduler, RandomResolutionScheduler
from scalemae.lib.transforms import CustomCompose
from scalemae.dataloaders.resic45 import build_resic
from scalemae.dataloaders.utils import get_dataset_and_sampler
from scalemae.eval.knn import kNN


Image.MAX_IMAGE_PIXELS = 1000000000
