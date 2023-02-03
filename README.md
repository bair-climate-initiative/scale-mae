# Scale-MAE 🛰️

## Requirements and Setup ##
Set up your Python (`conda` suggested) environment in the same way as the [MAE](github.com/facebookresearch/mae) repository. Please note that the `timm==0.3.2` patch is still needed for Scale-MAE.

In addition, install `GDAL`, `rasterio`, and `Shapely`.

## Pretraining ##
Datasets are defined by config files in `config` 

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

## Linear Probing ##

We will update codes for running linear probing soon.