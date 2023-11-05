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

## Data Preparation
Download the FMoW-rgb dataset as described in the [here](https://github.com/fMoW/dataset) and then make a symlink to the data directory in the root of this repo.  For example, if you downloaded the data to `~/data/fmow-rgb`, then run:

```bash
ln -s ~/data/fmow-rgb data
```

## Pretraining ##
Datasets are defined by config files in `config`.
```
# change to num of gpus you have
python -m torch.distributed.launch --nproc_per_node=4 \
    -m scalemae.main_pretrain
```

use `-h` to see details of all arguments. 


## Pretrained Models

* [**ViT Large 800 ep**](https://github.com/bair-climate-initiative/scale-mae/releases/download/base-800/scalemae-vitlarge-800.pth)



## Evaluation

### KNN Evaluation
```
python -m torch.distributed.launch --nproc_per_node=4 \
    -m scalemae.main_pretrain \
        --resume <path-to-model-checkpoint.pth> \
        --eval_only \
        --eval_dataset <eval_dataset_name>  \
        --eval_train_fnames <train_split_file>  \
        --eval_val_fnames <val_split_file>
```

We support resisc (default), airound, mlrsnet, and fmow kNN evaluation. We provide all split files in `splits` folder. If `--eval_train_fnames` and `--eval_val_fnames` are specified, the content of these two txt files will be read as the train split and test split. If this is the case, the root folder of the dataset is assumed to be the parent folder of such txt files. Alternatively, one can specify `--eval_path`. If this is the case, 90% of the data is randomly selected as the training set while the 10% is selected as the test set. The dataset is assumed to have the standard structure of `ImageFolder` in `torchvision`.  

### Finetuning

```
python -m torch.distributed.launch --nproc_per_node=4 \
    scalemae.main_linprobe \
        --checkpoint_path <path-to-model-checkpoint.pth>
```

Use the flag `--finetune` to enable full fine-tuning instead of a linear probing.

---

> Note: THIS SOFTWARE AND/OR DATA WAS DEPOSITED IN THE BAIR OPEN RESEARCH COMMONS REPOSITORY ON 2/8/23.



### Demo With DemoData


```bash

# Create demo train / vali data
DATA_PATH=$(python -m scalemae.demo)

echo "
data:
  type: ImageList
  length: 10
  img_dir: '$DATA_PATH'
  mean: [0.46921533, 0.46026663, 0.41329921]
  std: [0.1927, 0.1373, 0.1203]
  vis_factor: 1.0
" > $DATA_PATH/demo.yaml

cat  $DATA_PATH/demo.yaml


DEFAULT_ROOT_DIR=$HOME/exps/scalemae_demo

echo "
DEFAULT_ROOT_DIR      = $DEFAULT_ROOT_DIR
DATA_PATH             = $DATA_PATH
"


mkdir -p $DEFAULT_ROOT_DIR
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=1 --master_port=11085 -m scalemae.main_pretrain \
    --output_dir $DEFAULT_ROOT_DIR \
    --log_dir  $DEFAULT_ROOT_DIR \
    --config $DATA_PATH/demo.yaml \
    --eval_path "$DATA_PATH" \
    --batch_size 4 \
    --model mae_vit_base_patch16  \
    --mask_ratio 0.75 \
    --num_workers 0 \
    --epochs 300 \
    --target_size 224\
    --input_size 224\
    --self_attention\
    --scale_min 0.2 \
    --scale_max 1.0 \
    --warmup_epochs 40 \
    --blr 1.5e-4 --weight_decay 0.05 \
    --decoder_aux_loss_layers 1\
    --target_size_scheduler constant\
    --decoder_depth 8 \
    --no_autoresume \
    --use_mask_token \
    --skip_knn_eval \
    --fixed_output_size_min 224\
    --fixed_output_size_max 336\
    --absolute_scale 

    --loss_masking\
    --independent_fcn_head \
    --decoder_mode encoder\


```
