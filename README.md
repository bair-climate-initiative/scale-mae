# Scale-MAE 🛰️

![image](https://user-images.githubusercontent.com/1455579/217665789-b46d6830-445f-4151-b7a4-a2152a81a8d1.png)


This repository provides a reimplementation of the code for [Scale-MAE: A Scale-Aware Masked Autoencoder for Multiscale Geospatial Representation Learning](https://arxiv.org/abs/2212.14532) (the original code was optimized for our distributed cluster).

```
@article{reed2022scale,
  title={Scale-MAE: A Scale-Aware Masked Autoencoder for Multiscale Geospatial Representation Learning},
  author={Reed, Colorado J and Gupta, Ritwik and Li, Shufan and Brockman, Sarah and Funk, Christopher and Clipp, Brian and Candido, Salvatore and Uyttendaele, Matt and Darrell, Trevor},
  journal={arXiv preprint arXiv:2212.14532},
  year={2022}
}
```

* This repo is a modification on the [MAE repo](https://github.com/facebookresearch/mae). Installation and preparation follow that repo ;-).

* As mentioned in the MAE repo, this repo is based on [`timm==0.3.2`](https://github.com/rwightman/pytorch-image-models), for which a [fix](https://github.com/rwightman/pytorch-image-models/issues/420#issuecomment-776459842) is needed to work with PyTorch 1.8.1+. In addition, install gdal, rasterio, and Shapely.  This tends to work pretty well (but gdal is notoriously tricky):

## Installation
```bash
conda create -n scalemae python=3.9 geopandas # geopandas should install gdal correctly
conda activate scalemae
# replace with your desired pytorch target (e.g. cuda version)
conda install pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia
pip install -e .
```

Now shim timm (this is annoying and from the original MAE repo -- we'll try to fix at some point)

Open something like the following file (it'll return as the file that throws an error when you run `echo $(conda info --envs | grep scalemae | awk '{print $NF}')/lib/python3.9/site-packages/timm/models/layers/helpers.py`, and replace the following code:

```python
 from torch._six import container_abcs
```

With the following:
```python
import torch
TORCH_MAJOR = int(torch.__version__.split('.')[0])
TORCH_MINOR = int(torch.__version__.split('.')[1])

if TORCH_MAJOR == 1 and TORCH_MINOR < 8:
    from torch._six import container_abcs
else:
    import collections.abc as container_abcs
```

## Data Preparation
Download the FMoW-rgb dataset as described in the [here](https://github.com/fMoW/dataset).

## Pretraining ##
Datasets are defined by config files in `config` 


## Pretrained Models

* [**ViT Large 800 ep**](https://github.com/bair-climate-initiative/scale-mae/releases/download/base-800/scalemae-vitlarge-800.pth)
* [**ViT Base 800 ep**](https://github.com/bair-climate-initiative/scale-mae/releases/download/base-800/scalemae-vitbase-800.pth)





# TOD FIX ALL OF THIS

```
python -m torch.distributed.launch --nproc_per_node=4 --master_port=11083 main_pretrain.py\
    --batch_size 256 \
    --model mae_vit_base_patch16  \
    --mask_ratio 0.75 \
    --norm_pix_loss \
    --num_workers 20 \
    --epochs 300 \
    --target_size 448\
    --input_size 224\
    --self_attention\
    --scale_min 0.2 \
    --scale_max 1.0 \
    --output_dir <PATH> \
    --log_dir  <PATH> \
    --warmup_epochs 40 \
    --blr 1.5e-4 --weight_decay 0.05 \
    --config config/fmow-dgx.yaml \
    --decoder_aux_loss_layers 1
    --target_size_scheduler constant\
    --decoder_depth 2 \
    --no_autoresume \
    --use_mask_token \
    --loss_masking\
    --fixed_output_size_min 224\
    --fixed_output_size_max 336\
    --eval_train_fnames resisc45/train.txt \
    --eval_val_fnames resisc45/val.txt \
    --independent_fcn_head \
    --band_config 64 64 \
    --l1_loss_weight 0.75 \
    --base_resolution 2.5 \
    --use_l1_loss \
    $@ \
```

use `-h` to see more details of arguments. 

## Finetuning ##

```
python -m torch.distributed.launch --nproc_per_node=2 \
main_linprobe.py \
--model vit_large_patch16 \
--batch_size 256 \
--nb_classes 62 \
--epochs 100 \
--warmup_epochs 0 \
--input_size 224 \
--global_pool \
--blr 0.0003 \
--weight_decay 0.005 \
--config config/resisc-dgx.yaml \
--checkpoint_path <PATH> \
--eval_dataset resisc --eval_path resisc45/val.txt \
--name VANILLA-linprobe_resisc_blr3e-3-base1.0-gp --eval_base_resolution 1.0 --eval_scale 256 \
--finetune --linear_layer_scale 10 \
```

flag `--finetune` is enabled for fine-tuning and not enabled for linear probing


---

> Note: THIS SOFTWARE AND/OR DATA WAS DEPOSITED IN THE BAIR OPEN RESEARCH COMMONS REPOSITORY ON 2/8/23.
