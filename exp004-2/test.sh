#!/bin/bash

MODEL_PATH="./model"
if [[ ! -e $MODEL_PATH ]]; then
    echo "$MODEL_PATH not exist!"
    exit
fi

CUDA_VISIBLE_DEVICES=1 python main.py --test --init_model=./model/final.pth --test_epoch=10 --batch_size=64 --motion_range=4 --num_inputs=3 --num_channel=3 --test_dir=/home/yi/Downloads/mpii-test-64 2>&1 | tee $MODEL_PATH/test.log

cp test.sh $MODEL_PATH
