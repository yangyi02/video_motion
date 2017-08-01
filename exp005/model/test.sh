#!/bin/bash

MODEL_PATH="./model"
if [[ ! -e $MODEL_PATH ]]; then
    echo "$MODEL_PATH not exist!"
    exit
fi

python main.py --test --init_model=./model/final.pth --test_epoch=1 --batch_size=2 --motion_range=4 --num_inputs=2 --num_channel=3 --test_dir=/home/yi/Downloads/mpii-test-64 --display 2>&1 | tee $MODEL_PATH/test.log
# python main.py --test --init_model=./model/final.pth --test_epoch=1 --batch_size=2 --motion_range=4 --num_inputs=2 --num_channel=3 --test_dir=/home/yi/Downloads/mpii-64 --display 2>&1 | tee $MODEL_PATH/test.log

cp test.sh $MODEL_PATH
