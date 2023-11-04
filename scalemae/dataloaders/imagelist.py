import os
from typing import Any, Callable, Dict, List, Optional, Tuple, cast

from torchvision.datasets import VisionDataset


class ImageList(VisionDataset):
    """A generic data loader for a list of images in a text file"""

    def __init__(
        self,
        imglist_path: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        root = os.path.dirname(imglist_path)
        with open(imglist_path) as imglistr:
            self.imglist = [line.strip() for line in imglistr]
        super().__init__(root, transform=transform, target_transform=target_transform)

        classes, class_to_idx = self.find_classes(self.imglist)
        samples = [
            (os.path.join(root, fn), class_to_idx[self.filename_to_class(fn)])
            for fn in self.imglist
        ]
        print(classes)
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples
        self.targets = [s[1] for s in samples]

    def filename_to_class(self, fn: str) -> str:
        # hardcoded HACK that could break
        return os.path.dirname(fn).split("/")[1]

    def find_classes(self, filenames: List[str]) -> Tuple[List[str], Dict[str, int]]:
        """ """
        classes = sorted(list({self.filename_to_class(fn) for fn in filenames}))
        if len(classes) == 0:
            raise FileNotFoundError(f"Couldn't find any classes in filenames.")

        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = pil_loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self) -> int:
        return len(self.samples)


from PIL import Image


def pil_loader(path: str) -> Image.Image:
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")
