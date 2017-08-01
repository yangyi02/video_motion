#!/bin/bash

MODEL_PATH="./model"
if [[ ! -e $MODEL_PATH ]]; then
    echo "$MODEL_PATH not exist!"
    exit
fi

CUDA_VISIBLE_DEVICES=1 python main.py --test --batch_size=1 --init_model=./model/final.pth --test_epoch=1 --motion_range=5 --num_inputs=4 --image_size=64 --num_channel=3 --image_dir=/home/yi/Downloads/mpii-240-2 --display 
