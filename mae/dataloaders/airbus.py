import itertools
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from PIL import Image
from torch.utils.data import Dataset


class Airbus(Dataset):
    """Airbus Ship Detection dataset."""

    def __init__(self, root_dir, split_file, transform=None, classification=False):
        """
        Args:
            root_dir (string): Directory with all the images.
            split_file (string): File with COCO JSON for the split.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        if type(root_dir) is str:
            root_dir = Path(root_dir)

        self.root_dir = root_dir
        self.scenes = self._load_file_info(split_file)
        self.length = len(self.scenes)

        self.transform = transform
        self.classification = classification

    def _load_file_info(self, split_file):
        f = open(split_file)
        lines = f.readlines()
        # Skip header
        lines = lines[1:]
        pairs = [x.strip().split(",") for x in lines]

        # Remove corrupt files
        pairs = [x for x in pairs if x[0] != "6384c3e78.jpg"]
        grouped = [(k, list(v)) for k, v in itertools.groupby(pairs, lambda x: x[0])]

        return grouped

    def _rle2mask(self, mask_rle, shape=(768, 768)):
        """
        mask_rle: run-length as string formated (start length)
        shape: (width,height) of array to return
        Returns numpy array, 1 - mask, 0 - background
        """
        s = mask_rle.split()
        starts, lengths = (np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2]))
        starts -= 1
        ends = starts + lengths
        img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
        if len(mask_rle) == 0:
            return img.reshape(shape).T

        for lo, hi in zip(starts, ends):
            img[lo:hi] = 1
        return img.reshape(shape).T

    def _merge_masks(self, grouped_rle, shape=(768, 768)):
        mask = np.bitwise_or.reduce(
            [self._rle2mask(x[1], shape=shape) for x in grouped_rle]
        )
        return mask

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        ret = {}
        scene = self.scenes[idx]
        img = np.array(Image.open(self.root_dir / scene[0]), dtype=np.float32)
        img = np.transpose(img, (2, 0, 1))
        label = self._merge_masks(scene[1])

        label = np.expand_dims(label, axis=0)

        if self.transform:
            img = self.transform(img)
            label = transforms.ToTensor()(label)

            # TODO you must crop the label too!
            # i, j, h, w = transforms.RandomCrop.get_params(img, output_size=(128, 128))
            # img = F.crop(img, i, j, h, w)
            # label = F.crop(label, i, j, h, w)

        if self.classification:
            return img, 0

        return img, label
