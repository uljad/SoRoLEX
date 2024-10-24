#!/bin/bash
WANDB_API_KEY=$(cat ./docker/wandb_key)
git pull

script_and_args="${@:2}"
gpu=$1
echo "Launching container lstm_$gpu on GPU $gpu"
docker run \
    --env CUDA_VISIBLE_DEVICES=$gpu \
    --gpus all \
    -e WANDB_API_KEY=$WANDB_API_KEY \
    -v $(pwd):/home/duser/rl \
    --name lstm_$gpu \
    --user $(id -u) \
    -d \
    -t lstm \
    /bin/bash -c "$script_and_args"