# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
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
import wandb
from torch.utils.tensorboard import SummaryWriter

import os
import tempfile

import kornia.augmentation as K
import matplotlib.pyplot as plt
import models_mae
import numpy as np
import timm.optim.optim_factory as optim_factory
import util.misc as misc
import yaml
from dataloaders.resic45 import build_resic
from dataloaders.utils import get_dataset_and_sampler
from engine_pretrain import train_one_epoch
from eval.knn import kNN
from kornia.augmentation import AugmentationSequential
from kornia.constants import Resample
from lib.scheduler import ConstantResolutionScheduler, RandomResolutionScheduler
from lib.transforms import CustomCompose
from PIL import Image
from torch.distributed.elastic.multiprocessing.errors import record
from torch.utils.data import DataLoader, Subset
from torchgeo.datasets import NAIP, stack_samples
from torchgeo.datasets.utils import download_url
from torchgeo.samplers import RandomGeoSampler, Units
from torchvision import transforms
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from util.misc import is_main_process
from wandb_log import WANDB_LOG_IMG_CONFIG

Image.MAX_IMAGE_PIXELS = 1000000000
