#!/bin/bash

# Create directories if not exist
MODEL_PATH=./model
if [[ ! -e $MODEL_PATH ]]; then
    mkdir -p $MODEL_PATH
else
    echo "$MODEL_PATH already exist!"
    exit
fi

CUDA_VISIBLE_DEVICES=1 python main.py --train --method=unsupervised --train_epoch=100000 --test_interval=1000 --test_epoch=100 --learning_rate=0.001 --batch_size=64 --motion_range=5 --num_inputs=4 --image_size=64 --num_channel=3 --image_dir=/home/yi/Downloads/mpii-240-2 2>&1 | tee $MODEL_PATH/train.log

cp train.sh $MODEL_PATH/train.sh
