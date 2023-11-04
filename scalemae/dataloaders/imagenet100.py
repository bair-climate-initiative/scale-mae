import os

import torch
from torch.utils.data._utils.collate import default_collate
from torchvision import datasets


class ImageNet100Dataset(datasets.ImageFolder):
    def __init__(self, path, anno_file, transform) -> None:
        super().__init__(path, transform=transform)
        self.imgs = self.samples
        with open(anno_file) as f:
            files_100 = f.readlines()
        # breakpoint()
        files_100 = [x.replace("\n", "") for x in files_100]
        new_samples = []
        for x, y in self.samples:
            if any([t in x for t in files_100]):
                new_samples.append((x, y))
        self.samples = new_samples


def build_imagenet_sampler(config, num_replicas, rank, transforms):
    img_dir = config["data"]["img_dir"]
    dataset = ImageNet100Dataset(
        os.path.join(img_dir, "train"), anno_file="anno_100.txt", transform=transforms
    )
    sampler = torch.utils.data.DistributedSampler(
        dataset, num_replicas=num_replicas, rank=rank, shuffle=True
    )
    collate_fn = default_collate
    return dataset, sampler, collate_fn
