# rm -rf ./jobs/pretrain/*
export JOB_DIR=./jobs/pretrain
export IMAGENET_DIR=/shared/group/ilsvrc
export CUDA_VISIBLE_DEVICES=5
set -x
CUDA_VISIBLE_DEVICES=3 python -m torch.distributed.launch --nproc_per_node=1 --master_port=11085 main_pretrain.py\
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
    --output_dir /home/jacklishufan/exps/output_encoder_decoder_4\
    --log_dir  /home/jacklishufan/exps/output_encoder_decoder_4\
    --warmup_epochs 40 \
    --blr 1.5e-4 --weight_decay 0.05 \
    --config config/naip.yaml \
    --decoder_aux_loss_layers 1\
    --decoder_mode encoder\
    --target_size_scheduler constant\
    --decoder_depth 8 \
    --no_autoresume \
    --use_mask_token \
    --loss_masking\
    --skip_knn_eval \
    --fixed_output_size_min 224\
    --fixed_output_size_max 336\
    --eval_train_fnames /shared/jacklishufan/resisc45/train.txt\
    --eval_val_fnames /shared/jacklishufan/resisc45/val.txt \
    --independent_fcn_head \
    --absolute_scale \
    $@ \
    
    # --resume /shared/jacklishufan/mae/mae_visualize_vit_base.pth \
    # --restart \
    #    --use_mask_token \
# CUDA_VISIBLE_DEVICES=5 python -m torch.distributed.launch --nproc_per_node=1 main_pretrain.py\
#     --job_dir ${JOB_DIR} \
#     --nodes 1 \
#     --ngpus 1 \
#     --batch_size 4 \
#     --model mae_vit_base_patch16  \
#     --norm_pix_loss \
#     --mask_ratio 0.75 \
#     --epochs 100 \ 
#     --warmup_epochs 40 \
#     --blr 1.5e-4 --weight_decay 0.05 \
#     --data_path ${IMAGENET_DIR}

