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
python -m torch.distributed.launch --nproc_per_node=4
main_pretrain.py
```

use `-h` to see details of all arguments. 


## Pretrained Models

* [**ViT Large 800 ep**](https://github.com/bair-climate-initiative/scale-mae/releases/download/base-800/scalemae-vitlarge-800.pth)



## Evaluation

### KNN Evaluation
```
python -m torch.distributed.launch --nproc_per_node=4 \
main_pretrain.py \
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
main_linprobe.py \
--checkpoint_path <path-to-model-checkpoint.pth>
```

Use the flag `--finetune` to enable full fine-tuning instead of a linear probing.

---

> Note: THIS SOFTWARE AND/OR DATA WAS DEPOSITED IN THE BAIR OPEN RESEARCH COMMONS REPOSITORY ON 2/8/23.
