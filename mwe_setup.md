The following is a MWE to setup and run the repo. It will use docker to ensure
the example is reproducible.

We make the following hardware assumptions:

* At least 1 NVIDIA GPU


First we will build the base image.

```bash

docker run --runtime=nvidia -it $PYENV_IMAGE echo "hello world"
DOCKER_BUILDKIT=1 docker build --progress=plain \
    -t conda_base_image \
    -f ./dockerfiles/conda.Dockerfile .

```


Now create a container where this directory is mounted, and then start a shell.
It may help to mount your pip cache.

```bash

IMAGE_NAME=conda_base_image
docker run \
    --runtime=nvidia \
    --volume "$PWD":/io:ro \
    --volume "$HOME"/.cache/pip:/pip_cache \
    -it "$IMAGE_NAME" \
    /bin/bash 

```

Inside the docker shell, we now demonstrate a MWE of installing and running
this repo from scratch.

```bash

# Get a fresh copy of the repo inside the container
mkdir -p $HOME/code
# git clone /io/.git $HOME/code/scalemae
# or 
cp -rv /io $HOME/code/scalemae

# Navigate to the fresh copy
cd $HOME/code/scalemae


# Follow normal json instructions
# geopandas should install gdal correctly 

conda create -n scalemae python=3.9 geopandas -y

conda activate scalemae

conda install pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia -y

pip install -e .

pip install -r requirements/runtime.txt

pip install kwcoco[headless]

# Create demo train / vali data
DATA_PATH=$(python -m scalemae.demo)
# Run twice because the capture only works
# if the demo data is cached.
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
    --absolute_scale \ 
    --loss_masking \
    --independent_fcn_head \
    --decoder_mode encoder

```
