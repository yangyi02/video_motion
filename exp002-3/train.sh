#!/bin/bash

# Create directories if not exist
MODEL_PATH=./model
if [[ ! -e $MODEL_PATH ]]; then
    mkdir -p $MODEL_PATH
else
    echo "$MODEL_PATH already exist!"
    exit
fi

CUDA_VISIBLE_DEVICES=1 python main.py --train --method=unsupervised --train_epoch=10000 --test_interval=100 --test_epoch=10 --learning_rate=0.001 --batch_size=32 --motion_range=6 --num_inputs=1 --image_size=64 --num_channel=3 --image_dir=/home/yi/Downloads/mpii-64-one-2 2>&1 | tee $MODEL_PATH/train.log

cp train.sh $MODEL_PATH/train.sh
