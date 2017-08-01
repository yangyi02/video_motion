#!/bin/bash

MODEL_PATH="./model"
if [[ ! -e $MODEL_PATH ]]; then
    echo "$MODEL_PATH not exist!"
    exit
fi

CUDA_VISIBLE_DEVICES=0 python main.py --test --init_model=./model/final.pth --test_epoch=10 --batch_size=64 --motion_range=5 --num_inputs=4 --image_size=64 2>&1 | tee $MODEL_PATH/test.log

cp test.sh $MODEL_PATH
