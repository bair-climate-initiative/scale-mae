from kornia.augmentation import RandomGaussianBlur


class CustomCompose:
    def __init__(self, rescale_transform, other_transforms, src_transform):
        self.rescale_transform = rescale_transform
        self.other_transforms = other_transforms
        self.src_transform = src_transform

    def __call__(self, x, valid_masks=None):
        if valid_masks is not None:
            nodata = (x * (1 - valid_masks.float())).max()
        x_aug = self.rescale_transform(x)
        parms = self.rescale_transform._params
        # sanity check, comment if this is working
        # valid_masks = self.rescale_transform(valid_masks.float(), params=parms)
        # assert (x_aug==self.rescale_transform(x, params=parms)).all() #

        if valid_masks is not None:
            valid_masks = x_aug != nodata
            _, c, h, w = x_aug.shape
            zero_ratio = ((valid_masks == 0).sum((1, 2, 3)) / (h * w * c)).cpu().numpy()
        else:
            zero_ratio = -1

        if self.other_transforms:
            x_aug = self.other_transforms(x_aug)
        x_src = self.src_transform(x_aug)
        dx = parms["src"][:, 1, 0] - parms["src"][:, 0, 0]
        # dy = (parms['src'][:,2,1] - parms['src'][:,1,1])
        # assert (dx == dy).all()
        h, w = x_aug.shape[-2:]
        # assert h == w
        return x_aug, x_src, dx / h, zero_ratio, valid_masks


blur = RandomGaussianBlur((3, 3), (2.0, 2.0), p=0.5)


def get_inputs_outputs(img, res, target=None, target_res=None, strategy="naive"):
    # TODO: More strategies
    if target is not None:
        return img, res, target, target_res
    else:
        target = img
        target_res = res
        img = blur(img)
        return img, res, target, target_res
