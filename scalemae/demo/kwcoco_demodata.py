
def make_demo_image_folder():
    """
    Make data that agrees with :class:`torchvision.datasets.ImageFolder`.
    """
    import kwcoco
    import kwimage
    import ubelt as ub
    import rich

    config = {
        'backend': 'geowatch',
    }

    rich.print('config = {}'.format(ub.urepr(config, nl=1)))
    dpath = ub.Path.appdir('scalemae/tests/demo/imagefolder')
    stamp = ub.CacheStamp(fname='demo-images', dpath=dpath, depends=config)

    if stamp.expired():
        if config['backend'] == 'kwcoco':
            dset = kwcoco.CocoDataset.demo('vidshapes8')
            imwrite_kwargs = {}
        elif config['backend'] == 'geowatch':
            imwrite_kwargs = {
                'compress': 'DEFLATE',
                'blocksize': 128,
            }
            try:
                # geowatch >=0.13.x
                import geowatch
                dset = geowatch.coerce_kwcoco('geowatch', geodata=True)
            except Exception:
                # geowatch <=0.12.1
                import watch
                dset = watch.coerce_kwcoco('watch', geodata=True)

        for coco_img in ub.ProgIter(dset.images().coco_images, desc='setup images'):
            image_id = coco_img['id']

            img = coco_img.imdelay(channels='r|g|b')

            if imwrite_kwargs:
                from osgeo import osr
                # TODO: would be nice to have an easy to use mechanism to get
                # the gdal crs, probably one exists in pyproj.
                auth = coco_img['wld_crs_info']['auth']
                assert auth[0] == 'EPSG', 'unhandled auth'
                epsg = auth[1]
                axis_strat = getattr(osr, coco_img['wld_crs_info']['axis_mapping'])
                srs = osr.SpatialReference()
                srs.ImportFromEPSG(int(epsg))
                srs.SetAxisMappingStrategy(axis_strat)
                img_from_wld = kwimage.Affine.coerce(coco_img['wld_to_pxl'])
                wld_from_img = img_from_wld.inv()
                imwrite_kwargs['crs'] = srs.ExportToWkt()
                imwrite_kwargs['overviews'] = 2

            anns = dset.annots(image_id=image_id).objs
            for ann in anns:
                # Crop out the annotation
                annot_id = ann['id']
                box = kwimage.Box.coerce(ann['bbox'], format='xywh')
                sl = box.quantize().to_slice()
                crop = img.crop(sl, clip=False, wrap=False)

                resized = crop.resize((224, 224))
                resized = resized.optimize()
                crop_imdata = resized.finalize()

                # Get the transform that took us from the image to here.
                warps = resized.undo_warps(return_warps=True)
                assert len(warps[0]) == 1, 'should be r|g|b only'
                img_from_ann = warps[1][0]

                # Save in a folder with its category name
                category_name = dset.index.cats[ann['category_id']]['name']
                cat_dpath = (dpath / category_name).ensuredir()
                # crop_fpath = cat_dpath / f'annot_{annot_id:03d}.jpg'

                if imwrite_kwargs:
                    # Update new geotiff metadata based on where we cropped.
                    wld_from_asset = wld_from_img @ img_from_ann
                    imwrite_kwargs['transform'] = wld_from_asset

                crop_fpath = cat_dpath / f'annot_{annot_id:03d}.tif'
                kwimage.imwrite(crop_fpath, crop_imdata, **imwrite_kwargs)
        stamp.renew()
    return dpath
