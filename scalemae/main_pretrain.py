# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
"""
CommandLine:
    xdoctest -m /home/joncrall/code/watch/geowatch_tpl/submodules/scale-mae/scalemae/main_pretrain.py None
    python -m scalemae.main_pretrain --help

Example:
    >>> from scalemae.main_pretrain import *  # NOQA
"""
import argparse
import datetime
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import timm  # NOQA
import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as tv_transforms

import kornia.augmentation as K
import numpy as np  # NOQA

import timm.optim.optim_factory as optim_factory
import yaml
from kornia.augmentation import AugmentationSequential
from kornia.constants import Resample
from PIL import Image
from torch.distributed.elastic.multiprocessing.errors import record
from torch.utils.data import Subset

# from scalemae.lib.scheduler import ConstantResolutionScheduler, RandomResolutionScheduler
import scalemae.util.misc as misc
from scalemae import models_mae
from scalemae.lib.transforms import CustomCompose
from scalemae.dataloaders.utils import get_dataset_and_sampler, get_eval_dataset_and_transform
from scalemae.engine_pretrain import train_one_epoch
from scalemae.eval.knn import kNN
from scalemae.util.misc import NativeScalerWithGradNormCount as NativeScaler
from scalemae.util.resolution_sched import (
    get_output_size_scheduler,
    get_source_size_scheduler,
    get_target_size_scheduler,
)

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:
    SummaryWriter = None

try:
    import wandb
    from wandb_log import WANDB_LOG_IMG_CONFIG
except Exception:
    wandb = None
    WANDB_LOG_IMG_CONFIG = None

Image.MAX_IMAGE_PIXELS = 1000000000


def get_args_parser():
    parser = argparse.ArgumentParser("MAE pre-training", add_help=False)
    parser.add_argument(
        "--checkpoint_interval", default=20, type=int, help="How often to checkpoint"
    )
    parser.add_argument(
        "--batch_size",
        default=32,
        type=int,
        help="Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus",
    )
    parser.add_argument(
        "--knn", default=20, type=int, help="Number of neighbors to use for KNN"
    )

    parser.add_argument(
        "--knn_eval_freq",
        default=5,
        type=int,
        help="How often (epochs) to run knn eval",
    )

    parser.add_argument(
        "--skip_knn_eval",
        action="store_true",
        help="Skip kNN evaluation (for debug purposes, primarily)",
    )
    parser.set_defaults(skip_knn_eval=False)

    parser.add_argument(
        "--print_freq",
        default=20,
        type=int,
        help="How often (iters) print results to wandb",
    )

    parser.add_argument(
        "--eval_gsd",
        action="store_true",
        help="USE GSD Relative Embedding with base=224x224",
    )
    parser.add_argument(
        "--eval_base_resolution",
        default=1.0,
        type=float,
        help="Global Multiplication factor of Positional Embedding Resolution in KNN",
    )

    parser.add_argument("--epochs", default=800, type=int)
    parser.add_argument(
        "--accum_iter",
        default=1,
        type=int,
        help="Accumulate gradient iterations (for increasing the effective batch size under memory constraints)",
    )
    parser.add_argument("--config", default="config.yaml", type=str, help="Config file")
    parser.add_argument("--name", default="", type=str, help="Name of wandb entry")

    # Model parameters
    parser.add_argument(
        "--model",
        default="mae_vit_base_patch16",
        type=str,
        metavar="MODEL",
        help="Name of model to train",
    )

    # Model parameters
    parser.add_argument(
        "--wandb_id", default=None, type=str, help="Wandb id, useful for resuming runs"
    )

    parser.add_argument("--input_size", default=224, type=int, help="images input size")
    parser.add_argument(
        "--target_size", nargs="*", type=int, help="images input size", default=[448]
    )
    parser.add_argument(
        "--source_size", nargs="*", type=int, help="images source size", default=[224]
    )

    parser.add_argument(
        "--mask_ratio",
        default=0.75,
        type=float,
        help="Masking ratio (percentage of removed patches).",
    )

    parser.add_argument("--scale_min", default=0.2, type=float, help="Min RRC scale")

    parser.add_argument("--scale_max", default=1.0, type=float, help="Max RRC scale")

    parser.add_argument(
        "--norm_pix_loss",
        action="store_true",
        help="Use (per-patch) normalized pixels as targets for computing loss",
    )
    parser.add_argument(
        "--reconst_loss",
        action="store_false",
        dest="norm_pix_loss",
        help="Contrary to norm_pix_loss",
    )

    parser.add_argument(
        "--restart",
        action="store_true",
        help="Load the checkpoint, but start from epoch 0",
    )
    parser.set_defaults(norm_pix_loss=True)

    # Optimizer parameters
    parser.add_argument(
        "--weight_decay", type=float, default=0.05, help="weight decay (default: 0.05)"
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        metavar="LR",
        help="learning rate (absolute lr)",
    )
    parser.add_argument(
        "--blr",
        type=float,
        default=0.00015,
        metavar="LR",
        help="base learning rate: absolute_lr = base_lr * total_batch_size / 256",
    )
    parser.add_argument(
        "--min_lr",
        type=float,
        default=0.0,
        metavar="LR",
        help="lower lr bound for cyclic schedulers that hit 0",
    )

    parser.add_argument(
        "--warmup_epochs", type=int, default=20, metavar="N", help="epochs to warmup LR"
    )

    parser.add_argument(
        "--eval_path", default="resisc45", type=str, help="dataset path"
    )

    parser.add_argument(
        "--eval_train_fnames",
        default="resisc45/train.txt",
        type=str,
        help="dataset path",
    )
    parser.add_argument(
        "--eval_val_fnames",
        default="data/resisc45/val.txt",
        type=str,
        help="dataset path",
    )

    parser.add_argument(
        "--eval_dataset",
        default="resisc",
        type=str,
        help="name of eval dataset to use. Options are resisc (default), airound, mlrsnet, and fmow.",
    )

    parser.add_argument(
        "--eval_scale",
        nargs="*",
        default=[56, 112, 224],
        type=int,
        help="The scales at which to run evaluation for kNN",
    )

    parser.add_argument(
        "--output_dir",
        default="./output_dir",
        help="path where to save, empty for no saving",
    )
    parser.add_argument(
        "--log_dir", default="./output_dir", help="path where to tensorboard log"
    )
    parser.add_argument(
        "--device", default="cuda", help="device to use for training / testing"
    )
    parser.add_argument("--eval_only", action="store_true", help="Only do KNN Eval")
    parser.set_defaults(eval_only=False)
    parser.add_argument(
        "--no_autoresume",
        action="store_true",
        help="Dont autoresume from last checkpoint",
    )
    parser.set_defaults(no_autoresume=False)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--resume", default="", help="resume from checkpoint")

    parser.add_argument(
        "--start_epoch", default=0, type=int, metavar="N", help="start epoch"
    )
    parser.add_argument("--num_workers", default=10, type=int)
    parser.add_argument(
        "--pin_mem",
        action="store_true",
        help="Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.",
    )
    parser.add_argument("--no_pin_mem", action="store_false", dest="pin_mem")
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument(
        "--world_size", default=1, type=int, help="number of distributed processes"
    )
    parser.add_argument("--local_rank", default=-1, type=int)
    parser.add_argument("--dist_on_itp", action="store_true")
    parser.add_argument(
        "--dist_url", default="env://", help="url used to set up distributed training"
    )
    parser.add_argument(
        "--decoder_aux_loss_layers",
        default=1,
        type=int,
        help="number of decoder layers used in loss, 0 to use all layers",
    )

    parser.add_argument(
        "--fixed_output_size_min",
        default=224,
        type=int,
        help="if not 0, fix output dimension",
    )
    parser.add_argument(
        "--fixed_output_size_max",
        default=336,
        type=int,
        help="if not 0, fix output dimension",
    )

    parser.add_argument(
        "--decoder_depth",
        default=3,
        type=int,
        help="number of decoder layers used in loss, 0 to use all layers",
    )
    parser.add_argument(
        "--use_mask_token",
        action="store_true",
        help="If true, encoder receive tokens after standard demasking, if not, encoded patches are directly passed to decoder",
    )
    parser.add_argument(
        "--no_mask_token",
        action="store_false",
        dest="use_mask_token",
        help="Contrary to use_mask_token",
    )
    parser.set_defaults(use_mask_token=True)
    parser.add_argument(
        "--project_pos_emb",
        action="store_true",
        help="If true, adding a linear projection layer before the pos_emb is passed to decoder",
    )
    parser.add_argument(
        "--no_loss_masking",
        action="store_false",
        dest="loss_masking",
        help="If true, do not mask the loss for pixels that are not masked on input",
    )
    # self_attention
    parser.add_argument(
        "--self_attention", action="store_true", help="fake self attention"
    )
    # absolute_scale
    parser.add_argument(
        "--absolute_scale",
        action="store_true",
        help="Positional embedding is the same for each image (based on resolution)",
    )

    parser.add_argument(
        "--base_resolution",
        default=2.5,
        type=float,
        help="The base resolution to use for the period of the sin wave for positional embeddings",
    )

    parser.add_argument(
        "--target_size_scheduler",
        default="constant",
        type=str,
        help="Which target size to have at a certain step",
    )
    parser.add_argument(
        "--source_size_scheduler",
        default="constant",
        type=str,
        help="Which target size to have at a certain step",
    )
    parser.add_argument(
        "--fcn_dim", default=512, type=int, help="FCN Hidden Dimension "
    )
    parser.add_argument(
        "--fcn_layers", default=2, type=int, help="FCN Hidden Dimension "
    )
    parser.add_argument(
        "--share_fcn_head",
        action="store_false",
        dest="independent_fcn_head",
        help="Whether to use different decoder for two bands",
    )
    parser.add_argument(
        "--use_l1_loss",
        action="store_true",
        help="Whether to use different L1 loss for high frequency (encoder-gtp-fpn specific)",
    )
    parser.add_argument(
        "--band_config",
        nargs="*",
        type=int,
        default=[7, 56],
        help="list like [dim1, dim2]; Target High Freq = img - upsample(downsample(img,dim1)),Target Low Freq = upsample(downsample(img,dim2))",
    )
    parser.add_argument(
        "--l1_loss_weight",
        default=1.0,
        type=float,
        help="w,Weight of l1 loss, final loss is w * L_1_loss (high) + L_2_loss (low)",
    )
    parser.add_argument(
        "--progressive", action="store_true", help="Progressive upsample"
    )

    return parser


@record
def main(args):
    print("Starting pretrain")
    device = torch.device(args.device)
    misc.init_distributed_mode(args)
    num_tasks = misc.get_world_size()
    global_rank = misc.get_rank()
    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    ######## backwards compatability hacks
    if not isinstance(args.target_size, list):
        args.target_size = [args.target_size]

    if not isinstance(args.source_size, list):
        args.source_size = [args.source_size]
    ########################################

    if not args.eval_only:
        # Validate that all sizes in target_size are multiples of 16
        if len(args.target_size) > 0:
            assert all(
                [isinstance(i, int) for i in args.target_size]
            ), "Invalid multiscale input, it should be a json list of int, e.g. [224,448]"
            assert all(
                [i % 16 == 0 for i in args.target_size]
            ), "Decoder resolution must be a multiple of patch size (16)"

        # set a random wandb id before fixing random seeds
        if wandb is not None:
            random_wandb_id = wandb.util.generate_id()
        else:
            random_wandb_id = None

        print(f"job dir: {os.path.dirname(os.path.realpath(__file__))}")
        print(f"{args}".replace(", ", ",\n"))

        with open(args.config) as f:
            config = yaml.safe_load(f.read())
        args.data_config = config  # save on args so that it's prop'd to wandb
        if WANDB_LOG_IMG_CONFIG is not None:
            WANDB_LOG_IMG_CONFIG.mean = np.array(config["data"]["mean"])
            WANDB_LOG_IMG_CONFIG.std = np.array(config["data"]["std"])
            WANDB_LOG_IMG_CONFIG.factor = config["data"]["vis_factor"]

        if config["data"]["type"] in ["fmow"]:
            # We read in an image from PIL and crop an area twice the size of input size
            # transforms_train crops it down to a the proper target_size
            transforms_init = tv_transforms.Compose(
                [
                    tv_transforms.RandomCrop(args.input_size * 2, pad_if_needed=True),
                    tv_transforms.RandomHorizontalFlip(),
                    tv_transforms.ToTensor(),
                    tv_transforms.Normalize(
                        mean=config["data"]["mean"], std=config["data"]["std"]
                    ),
                ]
            )
            other_transforms = None
        else:
            transforms_init = None
            other_transforms = AugmentationSequential(
                K.RandomHorizontalFlip(),
                K.Normalize(mean=config["data"]["mean"], std=config["data"]["std"]),
            )

        # We will pass in the largest target_size to RRC
        target_size = max(args.target_size)
        transforms_train = CustomCompose(
            rescale_transform=K.RandomResizedCrop(
                (target_size, target_size),
                # (args.input_size, args.input_size),
                ratio=(1.0, 1.0),
                scale=(args.scale_min, args.scale_max),
                resample=Resample.BICUBIC.name,
            ),
            other_transforms=other_transforms,
            src_transform=K.Resize((args.input_size, args.input_size)),
        )

        dataset_train, sampler_train, train_collate = get_dataset_and_sampler(
            args,
            config,
            transforms=transforms_train,
            num_replicas=num_tasks,
            rank=global_rank,
            transforms_init=transforms_init,
        )

        if misc.is_main_process() and args.log_dir is not None:
            os.makedirs(args.log_dir, exist_ok=True)
            log_writer = SummaryWriter(log_dir=args.log_dir)
        else:
            log_writer = None

        batch_size_factor = config["data"].get("oversample", 1)
        data_loader_train = torch.utils.data.DataLoader(
            dataset_train,
            sampler=sampler_train,
            batch_size=int(args.batch_size * batch_size_factor),
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=True,
            collate_fn=train_collate,
        )

        output_size_scheduler = get_output_size_scheduler(args)
        target_size_scheduler = get_target_size_scheduler(args)
        source_size_scheduler = get_source_size_scheduler(args)

    ########################### EVAL SPECIFIC SETUP ###########################
    # backwards compatability so running runs dont break

    if hasattr(args, "eval_train_fnames"):
        dataset_eval_train, _ = get_eval_dataset_and_transform(
            args.eval_dataset, args.eval_train_fnames
        )
        dataset_eval_test, _ = get_eval_dataset_and_transform(
            args.eval_dataset, args.eval_val_fnames
        )
    elif hasattr(args, "eval_path"):
        dataset_eval, _ = get_eval_dataset_and_transform(
            args.eval_dataset, args.eval_path
        )
        n_eval = np.arange(len(dataset_eval))
        knn_train_size = int(0.9 * len(n_eval))
        knn_eval_size = int(0.1 * len(n_eval))
        np.random.shuffle(n_eval)
        idx_eval = n_eval
        idx_eval_train = idx_eval[:knn_train_size]
        idx_eval_test = idx_eval[knn_train_size : knn_train_size + knn_eval_size]
        dataset_eval_train = Subset(dataset_eval, idx_eval_train)
        dataset_eval_test = Subset(dataset_eval, idx_eval_test)

    print(f"Eval Dataset: {dataset_eval_train}")
    if args.eval_only:
        k_nn_batch_size = 64
    else:
        k_nn_batch_size = 32

    if args.distributed:
        sampler_eval_train = torch.utils.data.DistributedSampler(
            dataset_eval_train,
            num_replicas=num_tasks,
            rank=global_rank,
            shuffle=False,
            drop_last=False,
        )
        sampler_eval_test = torch.utils.data.DistributedSampler(
            dataset_eval_test,
            num_replicas=num_tasks,
            rank=global_rank,
            shuffle=False,
            drop_last=False,
        )
    else:
        print("Not using distributed sampler")
        sampler_eval_train = torch.utils.data.SequentialSampler(dataset_eval_train)
        sampler_eval_test = torch.utils.data.SequentialSampler(dataset_eval_test)

    data_loader_eval_train = torch.utils.data.DataLoader(
        dataset_eval_train,
        batch_size=k_nn_batch_size,
        sampler=sampler_eval_train,
        num_workers=args.num_workers,
    )
    data_loader_eval_test = torch.utils.data.DataLoader(
        dataset_eval_test,
        batch_size=k_nn_batch_size,
        sampler=sampler_eval_test,
        num_workers=args.num_workers,
    )
    ##########################################################################
    # define the model
    # handle fix size

    model = models_mae.__dict__[args.model](
        img_size=args.input_size,
        norm_pix_loss=args.norm_pix_loss,
        decoder_aux_loss_layers=args.decoder_aux_loss_layers,
        decoder_depth=args.decoder_depth,
        use_mask_token=args.use_mask_token,
        project_pos_emb=args.project_pos_emb,
        loss_masking=args.loss_masking,
        self_attention=args.self_attention,
        absolute_scale=args.absolute_scale,
        target_size=args.target_size,
        fixed_output_size=0,  # will be set dynamically online
        fcn_dim=args.fcn_dim,
        fcn_layers=args.fcn_layers,
        independent_fcn_head=args.independent_fcn_head,
        use_l1_loss=args.use_l1_loss,
        band_config=args.band_config,
        l1_loss_weight=args.l1_loss_weight,
        progressive=args.progressive,
    )

    model.to(device)

    model_without_ddp = model

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()

    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.gpu],
            find_unused_parameters=True,  # find_unused_parameters=True
        )
        model_without_ddp = model.module

    # following timm: set wd as 0 for bias and norm layers
    param_groups = optim_factory.param_groups_layer_decay(
        model_without_ddp, args.weight_decay
    )
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    loss_scaler = NativeScaler()
    if not args.resume and not args.no_autoresume:
        checkpoint_file = os.path.join(args.output_dir, "checkpoint-latest.pth")
        if os.path.exists(checkpoint_file):
            print(f"Resuming latest checkpoint from {checkpoint_file}")
            args.resume = checkpoint_file
    misc.load_model(
        args=args,
        model_without_ddp=model_without_ddp,
        optimizer=optimizer,
        loss_scaler=loss_scaler,
    )
    # state_dict = model.state_dict()
    # key = np.random.choice(list(state_dict.keys()))
    # state_dict[key].mean()
    # breakpoint()

    if misc.is_main_process() and not args.eval_only:
        if not args.wandb_id:
            args.wandb_id = random_wandb_id

        tag = "encoder-decoder"

        wandb_args = dict(
            project="multiscale_mae",
            entity="bair-climate-initiative",
            id=args.wandb_id,
            resume="allow",
            tags=[tag],
            config=args.__dict__,
        )
        if args.name:
            wandb_args.update(dict(name=args.name))

        wandb.init(**wandb_args)

        print(f"Start training for {args.epochs} epochs")
        print("Model = %s" % str(model_without_ddp))
        print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
        print("actual lr: %.2e" % args.lr)

        print("accumulate grad iterations: %d" % args.accum_iter)
        print("effective batch size: %d" % eff_batch_size)

    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs + 1):  # + 1 to do one final knn
        if (
            (epoch % args.knn_eval_freq == 0 or epoch == args.epochs) or args.eval_only
        ) and not args.skip_knn_eval:
            eval_res = {}
            for eval_scale in args.eval_scale:
                eval_res[eval_scale] = kNN(
                    cmd_args=args,
                    net=model,
                    trainloader=data_loader_eval_train,
                    testloader=data_loader_eval_test,
                    # TODO clean this
                    feat_dim=1024 if "large" in args.model else 768,
                    eval_scale=eval_scale,
                    eval_base_resolution=args.eval_base_resolution
                    if hasattr(args, "eval_base_resolution")
                    else 1.0,
                    gsd_embed=args.eval_gsd if hasattr(args, "eval_gsd") else False,
                )
                if misc.is_main_process():
                    print(f"eval results ({eval_scale}): {eval_res[eval_scale]}")
                    if not args.eval_only:
                        wandb.log(
                            {
                                f"knn-acc-{eval_scale}": eval_res[eval_scale] * 100.0,
                                "epoch": epoch,
                            }
                        )

            if args.eval_only or epoch == args.epochs:
                break

        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        train_stats = train_one_epoch(
            model,
            data_loader_train,
            optimizer,
            device,
            epoch,
            loss_scaler,
            log_writer=log_writer,
            args=args,
            scheduler=target_size_scheduler,
            source_size_scheduler=source_size_scheduler,
            fix_resolution_scheduler=output_size_scheduler,
        )
        if args.output_dir and (
            epoch % args.checkpoint_interval == 0 or epoch + 1 == args.epochs
        ):
            misc.save_model(
                args=args,
                model=model,
                model_without_ddp=model_without_ddp,
                optimizer=optimizer,
                loss_scaler=loss_scaler,
                epoch=epoch,
            )
        # always save the latest checkpoint (overwrites)
        misc.save_model(
            args=args,
            model=model,
            model_without_ddp=model_without_ddp,
            optimizer=optimizer,
            loss_scaler=loss_scaler,
            epoch=epoch,
            latest=True,
        )

        log_stats = {
            **{f"train_{k}": v for k, v in train_stats.items()},
            "epoch": epoch,
        }

        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(
                os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8"
            ) as f:
                f.write(json.dumps(log_stats) + "\n")
            wandb.log(log_stats)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"Training time {total_time_str}")
    if misc.is_main_process():
        wandb.finish()

    return eval_res


if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
    sys.exit(0)
