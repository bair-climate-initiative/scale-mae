import os

import torch
import scalemae.util.misc as misc  # NOQA
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder

from scalemae.dataloaders.airound import AIROUND_DATASET_STATS
from scalemae.dataloaders.cvbrct import CVBRCT_DATASET_STATS
from scalemae.dataloaders.eurosat import EUROSAT_DATASET_STATS
from scalemae.dataloaders.fmow import FMOW_DATASET_STATS, build_fmow
from scalemae.dataloaders.imagelist import ImageList
from scalemae.dataloaders.imagenet100 import build_imagenet_sampler
from scalemae.dataloaders.mlrsnet import MLRSNET_DATASET_STATS
from scalemae.dataloaders.naip import build_naip_sampler
from scalemae.dataloaders.optimal import OPTIMAL_DATASET_STATS
from scalemae.dataloaders.resic45 import RESIC_DATASET_STATS, build_resic
from scalemae.dataloaders.sentinel2 import build_sentinel_sampler
from scalemae.dataloaders.ucmerced import UCMERCED_DATASET_STATS
from scalemae.dataloaders.whurs import WHURS_DATASET_STATS
from scalemae.dataloaders.xview import build_xview2_sampler

dataset_stats_lookup = {
    "airound": AIROUND_DATASET_STATS,
    "cvbrct": CVBRCT_DATASET_STATS,
    "mlrsnet": MLRSNET_DATASET_STATS,
    "resisc": RESIC_DATASET_STATS,
    "eurosat": EUROSAT_DATASET_STATS,
    "optimal-31": OPTIMAL_DATASET_STATS,
    "whu-rs19": WHURS_DATASET_STATS,
    "ucmerced": UCMERCED_DATASET_STATS,
    "fmow": FMOW_DATASET_STATS,
}


def get_dataset_and_sampler(
    args,
    config,
    split="train",
    num_replicas=None,
    rank=None,
    transforms=None,
    transforms_init=None,
    linprobe_finetune=False,
):
    dataset_type = config["data"]["type"]
    if dataset_type == "NAIP":
        return build_naip_sampler(config, args, num_replicas, rank, transforms)
    elif dataset_type == "SENTINEL2":
        return build_sentinel_sampler(config, args, num_replicas, rank, transforms)
    elif dataset_type == "XView2":
        return build_xview2_sampler(
            config=config,
            num_replicas=num_replicas,
            rank=rank,
            transforms=transforms,
            split=split,
        )
    elif dataset_type == "ImageNet":
        return build_imagenet_sampler(
            config=config, num_replicas=num_replicas, rank=rank, transforms=transforms
        )
    elif dataset_type in ["fmow"]:
        dataset = datasets.ImageFolder(
            root=config["data"]["img_dir"],
            transform=transforms_init,
            is_valid_file=is_fmow_rgb,
        )
        sampler_train = torch.utils.data.DistributedSampler(
            dataset, num_replicas=num_replicas, rank=rank, shuffle=True
        )

        if not linprobe_finetune:
            return (
                dataset,
                sampler_train,
                TransformCollateFn(transforms, args.base_resolution),
            )
        else:
            return (
                dataset,
                sampler_train,
                TransformCollateFnLabel(transforms, args.base_resolution),
            )
    elif dataset_type == "resisc":
        dataset = build_resic(config["data"]["img_dir"], transforms=transforms_init)
        sampler_train = torch.utils.data.DistributedSampler(
            dataset, num_replicas=num_replicas, rank=rank, shuffle=True
        )
        if not linprobe_finetune:
            return (
                dataset,
                sampler_train,
                TransformCollateFn(transforms, args.base_resolution),
            )
        else:
            return (
                dataset,
                sampler_train,
                TransformCollateFnLabel(transforms, args.base_resolution),
            )
    elif dataset_type == "eurosat":
        dataset = datasets.ImageFolder(
            root=config["data"]["img_dir"], transform=transforms_init
        )
        sampler_train = torch.utils.data.DistributedSampler(
            dataset, num_replicas=num_replicas, rank=rank, shuffle=True
        )

        if not linprobe_finetune:
            return (
                dataset,
                sampler_train,
                TransformCollateFn(transforms, args.base_resolution),
            )
        else:
            return (
                dataset,
                sampler_train,
                TransformCollateFnLabel(transforms, args.base_resolution),
            )
    else:
        raise NotImplementedError


def is_fmow_rgb(fname: str) -> bool:
    return fname.endswith("_rgb.jpg")


class TransformCollateFn:
    def __init__(self, transforms, base_resolution=1.0):
        self.transforms = transforms
        self.base_resolution = base_resolution

    def __call__(self, samples):
        imgs = torch.stack(list(zip(*samples))[0])
        imgs, imgs_src, ratios, _, _ = self.transforms(imgs)
        res = ratios * self.base_resolution
        imgs_src_res = res * (imgs.shape[-1] / imgs_src.shape[-1])
        return (imgs_src, imgs_src_res, imgs, res), None


class TransformCollateFnLabel:
    def __init__(self, transforms, base_resolution=1.0):
        self.transforms = transforms
        self.base_resolution = base_resolution

    def __call__(self, samples):
        imgs = torch.stack(list(zip(*samples))[0])
        labels = torch.tensor([x[1] for x in samples])
        imgs, imgs_src, ratios, _, _ = self.transforms(imgs)
        res = ratios * self.base_resolution
        imgs_src_res = res * (imgs.shape[-1] / imgs_src.shape[-1])
        return (imgs_src, imgs_src_res, imgs, res, labels), None


def get_eval_dataset_and_transform(
    eval_dataset_id="resisc",
    eval_dataset_path="~/data/resisc",
    transforms_init=None,
    args=None,
):
    # All of these datasets are ImageFolders
    if eval_dataset_id in [
        "resisc",
        "mlrsnet",
        "airound",
        "cvbrct",
        "eurosat",
        "optimal-31",
        "whu-rs19",
        "ucmerced",
    ]:
        ds_stats = dataset_stats_lookup[eval_dataset_id]
        transform_normalize = transforms.Normalize(
            mean=ds_stats.PIXEL_MEANS, std=ds_stats.PIXEL_STD
        )
        use_transforms = [transforms.ToTensor(), transform_normalize]
        if transforms_init:
            use_transforms.insert(0, transforms_init)
        if eval_dataset_id == 'ucmerced':
            use_transforms.insert(0, transforms.Resize((256,256)))
        transform_eval = transforms.Compose(use_transforms)

        if os.path.isdir(eval_dataset_path):
            dataset_eval = ImageFolder(eval_dataset_path, transform=transform_eval)
        else:
            dataset_eval = ImageList(eval_dataset_path, transform=transform_eval)

    elif eval_dataset_id == "fmow":
        ds_stats = dataset_stats_lookup[eval_dataset_id]
        if transforms_init and args:
            transform_eval = transforms.Compose(
                [
                    # Resize only the short side
                    transforms.Resize(args.eval_scale),
                    # TODO this may not be the right thing to do here.
                    transforms.CenterCrop(args.eval_scale),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=ds_stats.PIXEL_MEANS, std=ds_stats.PIXEL_STD
                    ),
                ]
            )
        else:
            transform_eval = transforms.Compose(
                [
                    # TODO remove hardcoding px size?
                    transforms.Resize(512),  # downsample short side to 512
                    transforms.CenterCrop(512),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=ds_stats.PIXEL_MEANS, std=ds_stats.PIXEL_STD
                    ),
                ]
            )
        dataset_eval = build_fmow(eval_dataset_path, transforms=transform_eval)

    else:
        raise NotImplementedError

    return dataset_eval, transform_eval
