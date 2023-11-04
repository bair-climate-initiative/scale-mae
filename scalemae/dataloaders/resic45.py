import os

from torchvision import transforms
from torchvision.datasets import ImageFolder

from .imagelist import ImageList


class RESIC_DATASET_STATS:
    PIXEL_MEANS = [0.368, 0.381, 0.3436]
    PIXEL_STD = [0.2035, 0.1854, 0.1849]


def build_resic(data_root, transforms):
    # backwards compatable -- pass in a folder or a list of images
    # this hardcoding isn't great
    if os.path.isdir(data_root):
        return ImageFolder(data_root, transform=transforms)
    return ImageList(data_root, transforms)


def build_resic_gsd_resample(input_size=224, gsd_scale=1.0):
    if gsd_scale == 1.0:
        return []

    resample = transforms.Resize(size=(input_size * gsd_scale, input_size * gsd_scale))
    original = transforms.Resize(size=(input_size, input_size))

    return (resample, original)
