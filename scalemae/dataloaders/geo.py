import functools
import glob
import os
import re
import sys
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, cast

import numpy as np
import rasterio
import rasterio.merge
import torch
from rasterio.crs import CRS
from rasterio.io import DatasetReader
from rasterio.vrt import WarpedVRT
from rasterio.windows import from_bounds
from torch import Tensor
from torchgeo.datasets.geo import GeoDataset
from torchgeo.datasets.utils import BoundingBox, disambiguate_timestamp

from . import merge


class CustomRasterDataset(GeoDataset):
    """Abstract base class for :class:`GeoDataset` stored as raster files."""

    #: Glob expression used to search for files.
    #:
    #: This expression should be specific enough that it will not pick up files from
    #: other datasets. It should not include a file extension, as the dataset may be in
    #: a different file format than what it was originally downloaded as.
    filename_glob = "*"

    #: Regular expression used to extract date from filename.
    #:
    #: The expression should use named groups. The expression may contain any number of
    #: groups. The following groups are specifically searched for by the base class:
    #:
    #: * ``date``: used to calculate ``mint`` and ``maxt`` for ``index`` insertion
    #:
    #: When :attr:`separate_files`` is True, the following additional groups are
    #: searched for to find other files:
    #:
    #: * ``band``: replaced with requested band name
    filename_regex = ".*"

    #: Date format string used to parse date from filename.
    #:
    #: Not used if :attr:`filename_regex` does not contain a ``date`` group.
    date_format = "%Y%m%d"

    #: True if dataset contains imagery, False if dataset contains mask
    is_image = True

    #: True if data is stored in a separate file for each band, else False.
    separate_files = False

    #: Names of all available bands in the dataset
    all_bands: List[str] = []

    #: Names of RGB bands in the dataset, used for plotting
    rgb_bands: List[str] = []

    #: Color map for the dataset, used for plotting
    cmap: Dict[int, Tuple[int, int, int, int]] = {}

    def __init__(
        self,
        root: str = "data",
        crs: Optional[CRS] = None,
        res: Optional[float] = None,
        bands: Optional[Sequence[str]] = None,
        transforms: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
        cache: bool = True,
    ) -> None:
        """Initialize a new Dataset instance.
        Args:
            root: root directory where dataset can be found
            crs: :term:`coordinate reference system (CRS)` to warp to
                (defaults to the CRS of the first file found)
            res: resolution of the dataset in units of CRS
                (defaults to the resolution of the first file found)
            bands: bands to return (defaults to all bands)
            transforms: a function/transform that takes an input sample
                and returns a transformed version
            cache: if True, cache file handle to speed up repeated sampling
        Raises:
            FileNotFoundError: if no files are found in ``root``
        """
        super().__init__(transforms)

        self.root = root
        self.cache = cache
        self.nodata = None

        # Populate the dataset index
        i = 0
        pathname = os.path.join(root, "**", self.filename_glob)
        filename_regex = re.compile(self.filename_regex, re.VERBOSE)
        for filepath in glob.iglob(pathname, recursive=True):
            match = re.match(filename_regex, os.path.basename(filepath))
            if match is not None:
                try:
                    with rasterio.open(filepath) as src:
                        # See if file has a color map
                        if len(self.cmap) == 0:
                            try:
                                self.cmap = src.colormap(1)
                            except ValueError:
                                pass

                        if crs is None:
                            crs = src.crs
                        if res is None:
                            res = src.res[0]
                        if self.nodata == None:
                            self.nodata = src.nodata

                        with WarpedVRT(src, crs=crs) as vrt:
                            minx, miny, maxx, maxy = vrt.bounds
                except rasterio.errors.RasterioIOError:
                    # Skip files that rasterio is unable to read
                    continue
                else:
                    mint: float = 0
                    maxt: float = sys.maxsize
                    if "date" in match.groupdict():
                        date = match.group("date")
                        mint, maxt = disambiguate_timestamp(date, self.date_format)

                    coords = (minx, maxx, miny, maxy, mint, maxt)
                    self.index.insert(i, coords, filepath)
                    i += 1

        if i == 0:
            raise FileNotFoundError(
                f"No {self.__class__.__name__} data was found in '{root}'"
            )

        if bands and self.all_bands:
            band_indexes = [self.all_bands.index(i) + 1 for i in bands]
            self.bands = bands
            assert len(band_indexes) == len(self.bands)
        elif bands:
            msg = (
                f"{self.__class__.__name__} is missing an `all_bands` attribute,"
                " so `bands` cannot be specified."
            )
            raise AssertionError(msg)
        else:
            band_indexes = None
            self.bands = self.all_bands

        self.band_indexes = band_indexes
        self._crs = cast(CRS, crs)
        self.res = cast(float, res)

    def __getitem__(self, query: BoundingBox) -> Dict[str, Any]:
        """Retrieve image/mask and metadata indexed by query.
        Args:
            query: (minx, maxx, miny, maxy, mint, maxt) coordinates to index
        Returns:
            sample of image/mask and metadata at that index
        Raises:
            IndexError: if query is not found in the index
        """
        hits = self.index.intersection(tuple(query), objects=True)
        filepaths = [hit.object for hit in hits]

        if not filepaths:
            raise IndexError(
                f"query: {query} not found in index with bounds: {self.bounds}"
            )

        if self.separate_files:
            data_list: List[Tensor] = []
            valid_mask_list: List[Tensor] = []
            filename_regex = re.compile(self.filename_regex, re.VERBOSE)
            for band in self.bands:
                band_filepaths = []
                for filepath in filepaths:
                    filename = os.path.basename(filepath)
                    directory = os.path.dirname(filepath)
                    match = re.match(filename_regex, filename)
                    if match:
                        if "date" in match.groupdict():
                            start = match.start("band")
                            end = match.end("band")
                            filename = filename[:start] + band + filename[end:]
                    filepath = glob.glob(os.path.join(directory, filename))[0]
                    band_filepaths.append(filepath)
                data_list_internal, valid_mask_list_internal = self._merge_files(
                    band_filepaths, query
                )
                data_list.append(data_list_internal)
                valid_mask_list.append(valid_mask_list_internal)
            data = torch.cat(data_list)
            valid_mask = torch.cat(valid_mask_list)
        else:
            data, valid_mask = self._merge_files(filepaths, query, self.band_indexes)

        key = "image" if self.is_image else "mask"
        sample = {key: data, "crs": self.crs, "bbox": query, "validmask": valid_mask}

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    def _merge_files(
        self,
        filepaths: Sequence[str],
        query: BoundingBox,
        band_indexes: Optional[Sequence[int]] = None,
    ) -> Tensor:
        """Load and merge one or more files.
        Args:
            filepaths: one or more files to load and merge
            query: (minx, maxx, miny, maxy, mint, maxt) coordinates to index
            band_indexes: indexes of bands to be used
        Returns:
            image/mask at that index
        """
        if self.cache:
            vrt_fhs = [self._cached_load_warp_file(fp) for fp in filepaths]
        else:
            vrt_fhs = [self._load_warp_file(fp) for fp in filepaths]

        bounds = (query.minx, query.miny, query.maxx, query.maxy)
        if len(vrt_fhs) == 1:
            src = vrt_fhs[0]
            out_width = round((query.maxx - query.minx) / self.res)
            out_height = round((query.maxy - query.miny) / self.res)
            count = len(band_indexes) if band_indexes else src.count
            out_shape = (count, out_height, out_width)

            window = from_bounds(*bounds, src.transform)
            dest = src.read(indexes=band_indexes, out_shape=out_shape, window=window)
            valid_msk = dest != src.nodata
            # valid_msk = src.read_masks(
            #    indexes=band_indexes,
            #    out_shape=out_shape,
            #    window=window,
            # )
        else:
            dest, _, valid_msk = merge.merge(
                vrt_fhs, bounds, self.res, indexes=band_indexes
            )

        # fix numpy dtypes which are not supported by pytorch tensors
        if dest.dtype == np.uint16:
            dest = dest.astype(np.int32)
        elif dest.dtype == np.uint32:
            dest = dest.astype(np.int64)

        if valid_msk.dtype == np.uint16:
            valid_msk = valid_msk.astype(np.int32)
        elif valid_msk.dtype == np.uint32:
            valid_msk = valid_msk.astype(np.int64)

        tensor = torch.tensor(dest)
        msk_tensor = torch.tensor(valid_msk)

        assert tensor.shape == msk_tensor.shape

        return tensor, msk_tensor

    @functools.lru_cache(maxsize=128)
    def _cached_load_warp_file(self, filepath: str) -> DatasetReader:
        """Cached version of :meth:`_load_warp_file`.
        Args:
            filepath: file to load and warp
        Returns:
            file handle of warped VRT
        """
        return self._load_warp_file(filepath)

    def _load_warp_file(self, filepath: str) -> DatasetReader:
        """Load and warp a file to the correct CRS and resolution.
        Args:
            filepath: file to load and warp
        Returns:
            file handle of warped VRT
        """
        src = rasterio.open(filepath)

        # Only warp if necessary
        if src.crs != self.crs:
            vrt = WarpedVRT(src, crs=self.crs)
            src.close()
            return vrt
        else:
            return src
