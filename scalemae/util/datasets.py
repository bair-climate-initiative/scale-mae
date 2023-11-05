# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

import os

import PIL
from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torchvision import datasets, transforms


def make_demo_image_folder():
    import kwcoco
    import kwimage
    import ubelt as ub
    dpath = ub.Path.appdir('scalemae/tests/demo/imagefolder')
    stamp = ub.CacheStamp(fname='demo-images', dpath=dpath)
    if stamp.expired():
        dset = kwcoco.CocoDataset.demo('vidshapes8')
        for coco_img in ub.ProgIter(dset.images().coco_images, desc='setup images'):
            image_id = coco_img['id']

            img = coco_img.imdelay()

            anns = dset.annots(image_id=image_id).objs
            for ann in anns:
                # Crop out the annotation
                annot_id = ann['id']
                box = kwimage.Box.coerce(ann['bbox'], format='xywh')
                sl = box.quantize().to_slice()
                crop = img[sl]
                resized = crop.resize((224, 224))
                crop_imdata = resized.finalize()

                # Save in a folder with its category name
                category_name = dset.index.cats[ann['category_id']]['name']
                cat_dpath = (dpath / category_name).ensuredir()
                crop_fpath = cat_dpath / f'annot_{annot_id:03d}.jpg'
                kwimage.imwrite(crop_fpath, crop_imdata)
        stamp.renew()
    return dpath


def build_dataset(is_train, args, explicit_path=False):
    """
    Example:
        >>> # xdoctest: +REQUIRES(module:kwcoco)
        >>> # Hack: use a scriptconfig object to simulate args
        >>> from scalemae.util.datasets import *  # NOQA
        >>> import scriptconfig as scfg
        >>> data_path = make_demo_image_folder()
        >>> class DatasetArgs(scfg.DataConfig):
        >>>     data_path = None
        >>>     input_size = 224
        >>>     color_jitter = 0.1
        >>>     aa = 'rand-m9-mstd0.5-inc1'
        >>>     reprob = 0.25
        >>>     remode = 'pixel'
        >>>     recount = 1
        >>> args = DatasetArgs(data_path=data_path)
        >>> is_train = True
        >>> dataset = build_dataset(is_train, args, explicit_path=True)
        >>> for idx in range(len(dataset)):
        >>>     batch_item = dataset[idx]
    """

    # ---
    # JPC: This has a problem, there should have been a config passed here.
    # Not sure where it went but going to fudge it in.
    # original code:
    # transform = build_transform(is_train, args)
    # fudged:
    config = {
        'data': {'input_size': (args.input_size, args.input_size)},
    }
    transform = build_transform(is_train, args, config)
    # ---

    if explicit_path:
        root = args.data_path
    else:
        # AntiPattern: dont assume a directory structure unless you are
        # managing everything about that directory. For external data, give
        # the user full control to specify exactly a path is.
        # workaround: added explicit_path argument
        root = os.path.join(args.data_path, "train" if is_train else "val")

    dataset = datasets.ImageFolder(root, transform=transform)

    print(dataset)

    return dataset


def build_transform(is_train, args, config):
    mean = IMAGENET_DEFAULT_MEAN
    std = IMAGENET_DEFAULT_STD
    # train transform
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=config["data"]["input_size"],
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation="bicubic",
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            mean=mean,
            std=std,
        )
        return transform

    # eval transform
    t = []
    if config["data"]["input_size"] <= 224:
        crop_pct = 224 / 256
    else:
        crop_pct = 1.0
    size = int(config["data"]["input_size"] / crop_pct)
    t.append(
        transforms.Resize(
            size, interpolation=PIL.Image.BICUBIC
        )  # to maintain same ratio w.r.t. 224 images
    )
    t.append(transforms.CenterCrop(config["data"]["input_size"]))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)
