
def make_demo_image_folder():
    """
    Make data that agrees with :class:`torchvision.datasets.ImageFolder`.
    """
    import kwcoco
    import kwimage
    import ubelt as ub
    dpath = ub.Path.appdir('scalemae/tests/demo/imagefolder')
    stamp = ub.CacheStamp(fname='demo-images', dpath=dpath)
    if stamp.expired():
        dset = kwcoco.CocoDataset.demo('vidshapes8')
        for coco_img in ub.ProgIter(dset.images().coco_images, desc='setup images'):
            image_id = coco_img['id']

            img = coco_img.imdelay()

            anns = dset.annots(image_id=image_id).objs
            for ann in anns:
                # Crop out the annotation
                annot_id = ann['id']
                box = kwimage.Box.coerce(ann['bbox'], format='xywh')
                sl = box.quantize().to_slice()
                crop = img[sl]
                resized = crop.resize((224, 224))
                crop_imdata = resized.finalize()

                # Save in a folder with its category name
                category_name = dset.index.cats[ann['category_id']]['name']
                cat_dpath = (dpath / category_name).ensuredir()
                crop_fpath = cat_dpath / f'annot_{annot_id:03d}.jpg'
                kwimage.imwrite(crop_fpath, crop_imdata)
        stamp.renew()
    return dpath
