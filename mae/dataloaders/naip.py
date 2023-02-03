from typing import Any, Dict, Optional

from lib.transforms import get_inputs_outputs
from matplotlib import pyplot as plt
from samplers.distributed import DistributedRandomGeoSampler
from torchgeo.datasets import stack_samples
from torchgeo.samplers import Units

from .geo import CustomRasterDataset


class NAIP(CustomRasterDataset):
    """National Agriculture Imagery Program (NAIP) dataset.

    The `National Agriculture Imagery Program (NAIP)
    <https://www.fsa.usda.gov/programs-and-services/aerial-photography/imagery-programs/naip-imagery/>`_
    acquires aerial imagery during the agricultural growing seasons in the continental
    U.S. A primary goal of the NAIP program is to make digital ortho photography
    available to governmental agencies and the public within a year of acquisition.

    NAIP is administered by the USDA's Farm Service Agency (FSA) through the Aerial
    Photography Field Office in Salt Lake City. This "leaf-on" imagery is used as a base
    layer for GIS programs in FSA's County Service Centers, and is used to maintain the
    Common Land Unit (CLU) boundaries.

    If you use this dataset in your research, please cite it using the following format:

    * https://www.fisheries.noaa.gov/inport/item/49508/citation
    """

    # https://www.nrcs.usda.gov/Internet/FSE_DOCUMENTS/nrcs141p2_015644.pdf
    # https://planetarycomputer.microsoft.com/dataset/naip#Storage-Documentation
    filename_glob = "m_*.*"
    filename_regex = r"""
        ^m
        _(?P<quadrangle>\d+)
        _(?P<quarter_quad>[a-z]+)
        _(?P<utm_zone>\d+)
        _(?P<resolution>\d+)
        _(?P<date>\d+)
        (?:_(?P<processing_date>\d+))?
        \..*$
    """

    # Plotting
    all_bands = ["R", "G", "B", "NIR"]
    rgb_bands = ["R", "G", "B"]

    def plot(
        self,
        sample: Dict[str, Any],
        show_titles: bool = True,
        suptitle: Optional[str] = None,
    ) -> plt.Figure:
        """Plot a sample from the dataset.

        Args:
            sample: a sample returned by :meth:`RasterDataset.__getitem__`
            show_titles: flag indicating whether to show titles above each panel
            suptitle: optional string to use as a suptitle

        Returns:
            a matplotlib Figure with the rendered sample

        .. versionchanged:: 0.3
           Method now takes a sample dict, not a Tensor. Additionally, possible to
           show subplot titles and/or use a custom suptitle.
        """
        image = sample["image"][0:3, :, :].permute(1, 2, 0)

        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(4, 4))

        ax.imshow(image)
        ax.axis("off")
        if show_titles:
            ax.set_title("Image")

        if suptitle is not None:
            plt.suptitle(suptitle)

        return fig


from .geo import CustomRasterDataset


class NAIP(CustomRasterDataset):
    """National Agriculture Imagery Program (NAIP) dataset.

    The `National Agriculture Imagery Program (NAIP)
    <https://www.fsa.usda.gov/programs-and-services/aerial-photography/imagery-programs/naip-imagery/>`_
    acquires aerial imagery during the agricultural growing seasons in the continental
    U.S. A primary goal of the NAIP program is to make digital ortho photography
    available to governmental agencies and the public within a year of acquisition.

    NAIP is administered by the USDA's Farm Service Agency (FSA) through the Aerial
    Photography Field Office in Salt Lake City. This "leaf-on" imagery is used as a base
    layer for GIS programs in FSA's County Service Centers, and is used to maintain the
    Common Land Unit (CLU) boundaries.

    If you use this dataset in your research, please cite it using the following format:

    * https://www.fisheries.noaa.gov/inport/item/49508/citation
    """

    # https://www.nrcs.usda.gov/Internet/FSE_DOCUMENTS/nrcs141p2_015644.pdf
    # https://planetarycomputer.microsoft.com/dataset/naip#Storage-Documentation
    filename_glob = "m_*.*"
    filename_regex = r"""
        ^m
        _(?P<quadrangle>\d+)
        _(?P<quarter_quad>[a-z]+)
        _(?P<utm_zone>\d+)
        _(?P<resolution>\d+)
        _(?P<date>\d+)
        (?:_(?P<processing_date>\d+))?
        \..*$
    """

    # Plotting
    all_bands = ["R", "G", "B", "NIR"]
    rgb_bands = ["R", "G", "B"]

    def plot(
        self,
        sample: Dict[str, Any],
        show_titles: bool = True,
        suptitle: Optional[str] = None,
    ) -> plt.Figure:
        """Plot a sample from the dataset.

        Args:
            sample: a sample returned by :meth:`RasterDataset.__getitem__`
            show_titles: flag indicating whether to show titles above each panel
            suptitle: optional string to use as a suptitle

        Returns:
            a matplotlib Figure with the rendered sample

        .. versionchanged:: 0.3
           Method now takes a sample dict, not a Tensor. Additionally, possible to
           show subplot titles and/or use a custom suptitle.
        """
        image = sample["image"][0:3, :, :].permute(1, 2, 0)

        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(4, 4))

        ax.imshow(image)
        ax.axis("off")
        if show_titles:
            ax.set_title("Image")

        if suptitle is not None:
            plt.suptitle(suptitle)

        return fig


class NAIPStackSampleCollateFn:
    def __init__(self, transforms, base_resolution=1.0):
        self.transforms = transforms
        self.base_resolution = base_resolution

    def __call__(self, samples):
        targets = stack_samples(samples)["image"][:, :3, :, :]
        valid_masks = stack_samples(samples)["validmask"][:, :3, :, :]
        targets = targets.float() / 255
        if self.transforms is not None:
            targets, imgs_src, ratios, zero_ratio, valid_masks = self.transforms(
                targets, valid_masks
            )
            targets_res = ratios * self.base_resolution
            imgs_src_res = targets_res * (targets.shape[-1] / imgs_src.shape[-1])
        return get_inputs_outputs(imgs_src, imgs_src_res, targets, targets_res), dict(
            zero_ratio=zero_ratio, valid_masks=valid_masks
        )


def build_naip_sampler(config, args, num_replicas, rank, transforms):
    config = config["data"]
    naip = NAIP(config["img_dir"])

    # To support multiple output sizes per batch, TorchGeo should crop to the largest possible target size first
    sampler = DistributedRandomGeoSampler(
        naip,
        size=1024,
        length=config["length"],
        units=Units.PIXELS,
        num_replicas=num_replicas,
        rank=rank,
    )
    collate_fn = NAIPStackSampleCollateFn(transforms, args.base_resolution)
    return naip, sampler, collate_fn
