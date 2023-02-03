import torch
from lib.transforms import get_inputs_outputs
from samplers.distributed import DistributedRandomGeoSampler
from torchgeo.datasets import RasterDataset, stack_samples
from torchgeo.samplers import Units

from .geo import CustomRasterDataset


class Sentinel2StackSampleCollateFn:
    def __init__(self, transforms, over_sample_factor=1.0, base_resolution=1.0):
        self.transforms = transforms
        self.over_sample_factor = over_sample_factor
        self.base_resolution = base_resolution

    def __call__(self, samples):
        imgs = stack_samples(samples)["image"][:, :3, :, :]
        valid_masks = stack_samples(samples)["validmask"]
        b, c, h, w = imgs.shape
        tgt_b = int(b / self.over_sample_factor)
        zero_ratio = (valid_masks == 0).sum((1, 2, 3)) / (h * w * c)
        zero_ratio_order = torch.argsort(zero_ratio, descending=False)
        imgs = imgs[zero_ratio_order][:tgt_b].contiguous()
        valid_masks = valid_masks[zero_ratio_order][:tgt_b].contiguous()
        assert imgs.shape == (tgt_b, c, h, w)
        imgs = imgs.float()  # / 255
        if self.transforms is not None:
            imgs, imgs_src, ratios, zero_ratio, valid_masks = self.transforms(
                imgs, valid_masks
            )  # ratio is crop_dim / original_dim, so resolution should be 1/ ratios
            res = ratios * self.base_resolution
            imgs_src_res = res * (imgs.shape[-1] / imgs_src.shape[-1])
        return get_inputs_outputs(imgs_src, imgs_src_res, imgs, res), dict(
            zero_ratio=zero_ratio, valid_masks=valid_masks
        )


class Sentinel2(CustomRasterDataset):
    filename_glob = "T*_B02_10m.tif"
    filename_regex = r"^.{6}_(?P<date>\d{8}T\d{6})_(?P<band>B0[\d])"
    date_format = "%Y%m%dT%H%M%S"
    is_image = True
    separate_files = True
    all_bands = ["B02", "B03", "B04"]
    rgb_bands = ["B04", "B03", "B02"]


def build_sentinel_sampler(config, args, num_replicas, rank, transforms):
    config = config["data"]
    dataset = Sentinel2(config["img_dir"])
    over_sample_factor = config["oversample"]
    sampler = DistributedRandomGeoSampler(
        dataset,
        size=config["size"],
        length=int(config["length"] * over_sample_factor),
        units=Units.PIXELS,
        num_replicas=num_replicas,
        rank=rank,
    )
    collate_fn = Sentinel2StackSampleCollateFn(
        transforms, over_sample_factor, args.base_resolution
    )
    return dataset, sampler, collate_fn
