import os

from torchvision.datasets import ImageFolder

from .imagelist import ImageList


class FMOW_DATASET_STATS:
    PIXEL_MEANS = [0.485, 0.456, 0.406]
    PIXEL_STD = [0.229, 0.224, 0.225]


def is_fmow_rgb(fname: str) -> bool:
    return fname.endswith("_rgb.jpg")


def build_fmow(data_root, transforms):
    if os.path.isdir(data_root):
        return ImageFolder(
            root=data_root, transform=transforms, is_valid_file=is_fmow_rgb
        )
    return ImageList(data_root, transforms)
