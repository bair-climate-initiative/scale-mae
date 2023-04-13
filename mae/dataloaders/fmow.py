import os

from torchvision.datasets import ImageFolder
from .imagelist import pil_loader
from .imagelist import ImageList
import json


class FMOW_DATASET_STATS:
    PIXEL_MEANS = [0.485, 0.456, 0.406]
    PIXEL_STD = [0.229, 0.224, 0.225]


def is_fmow_rgb(fname: str) -> bool:
    return fname.endswith("_rgb.jpg")


class ImageListWithGSD(ImageList):
    """A generic data loader for a list of images in a text file"""

    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        json_path = path.replace(".jpg", ".json")
        with open(json_path) as f:
            json_data = json.loads(f.read())
        sample = pil_loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        target = {"target": target, "gsd": json_data["pan_resolution_dbl"]}

        return sample, target

    def __len__(self) -> int:
        return len(self.samples)


class ImageFolderWithGSD(ImageFolder):
    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        json_path = path.replace(".jpg", ".json")
        with open(json_path) as f:
            json_data = json.loads(f.read())
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        target = {"target": target, "gsd": json_data["pan_resolution_dbl"]}
        return sample, target


def build_fmow(data_root, transforms):
    if os.path.isdir(data_root):
        return ImageFolder(
            root=data_root, transform=transforms, is_valid_file=is_fmow_rgb
        )
    return ImageList(data_root, transforms)
