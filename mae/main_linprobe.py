# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# MoCo v3: https://github.com/facebookresearch/moco-v3
# --------------------------------------------------------

import argparse
import datetime
import json
import os
import time
from pathlib import Path

import kornia.augmentation as K
import numpy as np
import timm
import timm.optim.optim_factory as optim_factory
import torch
import torch.backends.cudnn as cudnn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.transforms as tv_transforms
import wandb
import yaml
from kornia.augmentation import AugmentationSequential
from kornia.constants import Resample
from torch.utils.data import Subset
from torch.utils.tensorboard import SummaryWriter
from wandb_log import WANDB_LOG_IMG_CONFIG

import models_vit
import util.lr_decay as lrd
import util.misc as misc
from dataloaders.utils import get_dataset_and_sampler, get_eval_dataset_and_transform
from engine_finetune import evaluate, train_one_epoch
from lib.transforms import CustomCompose
from PIL import Image
from timm.models.layers import trunc_normal_
from util.lars import LARS
from util.misc import NativeScalerWithGradNormCount as NativeScaler

Image.MAX_IMAGE_PIXELS = 1000000000


def get_args_parser():
    parser = argparse.ArgumentParser(
        "MAE linear probing for image classification", add_help=False
    )

    parser.add_argument(
        "--checkpoint_interval", default=20, type=int, help="How often to checkpoint"
    )
    parser.add_argument(
        "--batch_size",
        default=64,
        type=int,
        help="Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus",
    )

    parser.add_argument(
        "--print_freq",
        default=20,
        type=int,
        help="How often (iters) print results to wandb",
    )
    parser.add_argument(
        "--drop_path",
        type=float,
        default=0.1,
        metavar="PCT",
        help="Drop path rate (default: 0.1)",
    )
    parser.add_argument("--epochs", default=400, type=int)
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

    parser.add_argument("--linear_layer_scale", default=1.0, type=float, help="")

    # Model parameters
    parser.add_argument(
        "--wandb_id", default=None, type=str, help="Wandb id, useful for resuming runs"
    )

    parser.add_argument("--input_size", default=224, type=int, help="images input size")
    parser.add_argument(
        "--target_size", nargs="*", type=int, help="images input size", default=[224]
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

    parser.add_argument("--scale_min", default=0.5, type=float, help="Min RRC scale")

    parser.add_argument("--scale_max", default=1.0, type=float, help="Max RRC scale")

    parser.add_argument(
        "--norm_pix_loss",
        action="store_true",
        help="Use (per-patch) normalized pixels as targets for computing loss",
    )

    parser.add_argument(
        "--restart",
        action="store_true",
        help="Load the checkpoint, but start from epoch 0",
    )
    parser.set_defaults(norm_pix_loss=False)

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
        default=1e-3,
        metavar="LR",
        help="base learning rate: absolute_lr = base_lr * total_batch_size / 256",
    )
    parser.add_argument(
        "--layer_decay",
        type=float,
        default=0.75,
        help="layer-wise lr decay from ELECTRA/BEiT",
    )
    parser.add_argument(
        "--min_lr",
        type=float,
        default=0.0,
        metavar="LR",
        help="lower lr bound for cyclic schedulers that hit 0",
    )

    parser.add_argument(
        "--warmup_epochs", type=int, default=0, metavar="N", help="epochs to warmup LR"
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
    parser.add_argument(
        "--eval_dataset",
        default="resisc",
        type=str,
        help="name of eval dataset to use. Options are resisc (default), airound, mlrsnet, and fmow.",
    )
    parser.add_argument(
        "--eval_path", default="resisc45", type=str, help="dataset path"
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
    parser.add_argument(
        "--eval_reference_resolution",
        default=224,
        type=float,
        help="Reference input resolution to scale GSD factor by in eval.",
    )
    parser.add_argument(
        "--eval_scale", default=224, type=int, help="The size of the eval input."
    )
    parser.add_argument("--eval", action="store_true", help="Perform evaluation only")
    parser.add_argument(
        "--dist_eval",
        action="store_true",
        default=False,
        help="Enabling distributed evaluation (recommended during training for faster monitor",
    )
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
        "--base_resolution",
        default=2.5,
        type=float,
        help="The base resolution to use for the period of the sin wave for positional embeddings",
    )

    # * Finetuning params
    parser.add_argument(
        "--finetune",
        action="store_true",
        help="If true, finetune. If false, linear probe.",
    )
    parser.add_argument(
        "--checkpoint_path", default=None, type=str, help="Path to checkpoint weights."
    )
    parser.add_argument("--global_pool", action="store_true")
    parser.set_defaults(global_pool=False)
    parser.add_argument(
        "--cls_token",
        action="store_false",
        dest="global_pool",
        help="Use class token instead of global pool for classification",
    )

    parser.add_argument(
        "--nb_classes",
        default=1000,
        type=int,
        help="number of the classification types",
    )

    return parser


def main(args):
    misc.init_distributed_mode(args)
    num_tasks = misc.get_world_size()
    global_rank = misc.get_rank()

    print(f"job dir: {os.path.dirname(os.path.realpath(__file__))}")
    print(f"{args}".replace(", ", ",\n"))

    device = torch.device(args.device)

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

    # Validate that all sizes in target_size are multiples of 16
    if len(args.target_size) > 0:
        assert all(
            [type(i) == int for i in args.target_size]
        ), "Invalid multiscale input, it should be a json list of int, e.g. [224,448]"
        assert all(
            [i % 16 == 0 for i in args.target_size]
        ), "Decoder resolution must be a multiple of patch size (16)"

    # set a random wandb id before fixing random seeds
    random_wandb_id = wandb.util.generate_id()

    print(f"job dir: {os.path.dirname(os.path.realpath(__file__))}")
    print(f"{args}".replace(", ", ",\n"))

    with open(args.config) as f:
        config = yaml.safe_load(f.read())
    args.data_config = config  # save on args so that it's prop'd to wandb

    if config["data"]["type"] in ["fmow"]:
        # We read in an image from PIL and crop an area twice the size of input size
        # transforms_train crops it down to a the proper target_size
        transforms_init = tv_transforms.Compose(
            [
                # tv_transforms.RandomCrop(args.input_size * 2, pad_if_needed=True),
                # tv_transforms.Resize((args.input_size, args.input_size)),
                tv_transforms.Resize(args.input_size),
                # tv_transforms.RandomResizedCrop(args.input_size, scale=(0.2, 1.0), interpolation=3),
                tv_transforms.RandomCrop(args.input_size),
                tv_transforms.RandomHorizontalFlip(),
                tv_transforms.ToTensor(),
                tv_transforms.Normalize(
                    mean=config["data"]["mean"], std=config["data"]["std"]
                ),
            ]
        )
        other_transforms = None
    else:
        transforms_init = tv_transforms.ToTensor()
        other_transforms = AugmentationSequential(
            K.RandomHorizontalFlip(),
            K.Normalize(mean=config["data"]["mean"], std=config["data"]["std"]),
        )

    # We will pass in the largest target_size to RRC
    target_size = max(args.target_size)
    transforms_train = CustomCompose(
        rescale_transform=K.RandomResizedCrop(
            (args.input_size, args.input_size),
            ratio=(1.0, 1.0),
            scale=(args.scale_min, args.scale_max),
            resample=Resample.BICUBIC.name,
        ),
        other_transforms=other_transforms,
        src_transform=K.Resize((args.input_size, args.input_size)),
    )

    transforms_val_init = tv_transforms.Resize((args.eval_scale, args.eval_scale))

    dataset_train, sampler_train, train_collate = get_dataset_and_sampler(
        args,
        config,
        transforms=transforms_train,
        num_replicas=num_tasks,
        rank=global_rank,
        transforms_init=transforms_init,
        linprobe_finetune=True,
    )

    dataset_val, transforms_val = get_eval_dataset_and_transform(
        args.eval_dataset,
        args.eval_path,
        transforms_init=transforms_val_init,
        args=args,
    )
    sampler_val = torch.utils.data.DistributedSampler(
        dataset_val,
        num_replicas=num_tasks,
        rank=global_rank,
        shuffle=False,
        drop_last=False,
    )

    print(dataset_train)
    print(dataset_val)

    if global_rank == 0 and args.log_dir is not None and not args.eval:
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

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=args.batch_size,
        sampler=sampler_val,
        num_workers=args.num_workers,
    )

    model = models_vit.__dict__[args.model](
        img_size=args.input_size,
        num_classes=args.nb_classes,
        global_pool=args.global_pool,
    )

    if args.checkpoint_path:
        checkpoint = torch.load(args.checkpoint_path, map_location="cpu")

        print(f"Load pre-trained checkpoint from: {args.checkpoint_path}")
        checkpoint_model = checkpoint["model"]
        state_dict = model.state_dict()
        for k in ["head.weight", "head.bias"]:
            if (
                k in checkpoint_model
                and checkpoint_model[k].shape != state_dict[k].shape
            ):
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]
        if args.input_size != 224:
            if (
                "pos_embed" in checkpoint_model
                and checkpoint_model["pos_embed"].shape != state_dict["pos_embed"].shape
            ):
                print(f"Removing key pos_embed from pretrained checkpoint")
                del checkpoint_model["pos_embed"]

        # interpolate position embedding
        # We do not do this in Scale-MAE since we use a resolution-specific
        # pos embedding in forward_features
        # interpolate_pos_embed(model, checkpoint_model)

        # load pre-trained model
        msg = model.load_state_dict(checkpoint_model, strict=False)
        print(msg)

        if args.global_pool:
            assert set(msg.missing_keys) == {
                "head.weight",
                "head.bias",
                "fc_norm.weight",
                "fc_norm.bias",
            }
        else:
            if args.input_size != 224:
                assert set(msg.missing_keys) == {
                    "head.weight",
                    "head.bias",
                    "pos_embed",
                }
            else:
                assert set(msg.missing_keys) == {"head.weight", "head.bias"}

        if not args.eval:
            # manually initialize fc layer: following MoCo v3
            trunc_normal_(model.head.weight, std=0.01)
            # model.head.bias.data.zero_()

    # for linear prob only
    # hack: revise model's head with BN
    model.head = torch.nn.Sequential(
        torch.nn.BatchNorm1d(model.head.in_features, affine=False, eps=1e-6), model.head
    )

    if not args.finetune:
        # Linear probe
        # freeze all but the head
        for _, p in model.named_parameters():
            p.requires_grad = False
        for _, p in model.head.named_parameters():
            p.requires_grad = True
    else:
        for _, p in model.named_parameters():
            p.requires_grad = True
        for _, p in model.head.named_parameters():
            p.requires_grad = True

    # ScaleMAE does not use the pos_embed within ViT
    model.pos_embed.requires_grad = False

    model.to(device)

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()

    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    if not args.finetune:
        # Linear probe
        param_groups = optim_factory.param_groups_layer_decay(
            model_without_ddp, args.weight_decay
        )
    else:
        # build optimizer with layer-wise lr decay (lrd)
        param_groups = lrd.param_groups_lrd(
            model_without_ddp,
            args.weight_decay,
            no_weight_decay_list=model_without_ddp.no_weight_decay(),
            layer_decay=args.layer_decay,
        )
        param_groups[-1]["lr_scale"] *= args.linear_layer_scale
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    print(optimizer)
    loss_scaler = NativeScaler()

    criterion = torch.nn.CrossEntropyLoss()

    print("criterion = %s" % str(criterion))

    misc.load_model(
        args=args,
        model_without_ddp=model_without_ddp,
        optimizer=optimizer,
        loss_scaler=loss_scaler,
    )

    if args.eval:
        test_stats = evaluate(
            data_loader_val,
            model,
            device,
            eval_base_resolution=args.eval_base_resolution,
            gsd_embed=args.eval_gsd,
            eval_scale=args.eval_scale,
            reference_size=args.eval_reference_resolution,
        )
        print(
            f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%"
        )
        exit(0)

    if misc.is_main_process() and not args.eval:
        if not args.wandb_id:
            args.wandb_id = random_wandb_id

        wandb_args = dict(
            project="fmow-ft",
            entity="bair-climate-initiative",
            resume="allow",
            config=args.__dict__,
        )
        if args.name:
            wandb_args.update(dict(name=args.name))

        wandb.init(**wandb_args)

        if not args.finetune:
            print(f"Start linear probing for {args.epochs} epochs")
        else:
            print(f"Start finetuning for {args.epochs} epochs")
        print("Model = %s" % str(model_without_ddp))
        print(f"number of params (M): {(n_parameters / 1.e6):.2f}")
        print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
        print("actual lr: %.2e" % args.lr)

        print("accumulate grad iterations: %d" % args.accum_iter)
        print("effective batch size: %d" % eff_batch_size)

    start_time = time.time()
    max_accuracy = 0.0
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        train_stats = train_one_epoch(
            model,
            criterion,
            data_loader_train,
            optimizer,
            device,
            epoch,
            loss_scaler,
            max_norm=None,
            log_writer=log_writer,
            args=args,
        )
        if args.output_dir:
            misc.save_model(
                args=args,
                model=model,
                model_without_ddp=model_without_ddp,
                optimizer=optimizer,
                loss_scaler=loss_scaler,
                epoch=epoch,
            )

        test_stats = evaluate(
            data_loader_val,
            model,
            device,
            eval_base_resolution=args.eval_base_resolution,
            gsd_embed=args.eval_gsd,
            eval_scale=args.eval_scale,
            reference_size=args.eval_reference_resolution,
        )
        print(
            f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%"
        )
        max_accuracy = max(max_accuracy, test_stats["acc1"])
        print(f"Max accuracy: {max_accuracy:.2f}%")

        if log_writer is not None:
            log_writer.add_scalar("perf/test_acc1", test_stats["acc1"], epoch)
            log_writer.add_scalar("perf/test_acc5", test_stats["acc5"], epoch)
            log_writer.add_scalar("perf/test_loss", test_stats["loss"], epoch)

        log_stats = {
            **{f"train_{k}": v for k, v in train_stats.items()},
            **{f"test_{k}": v for k, v in test_stats.items()},
            "epoch": epoch,
            "n_parameters": n_parameters,
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


if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
