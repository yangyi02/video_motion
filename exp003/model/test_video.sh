#!/bin/bash

MODEL_PATH="./model"
if [[ ! -e $MODEL_PATH ]]; then
    echo "$MODEL_PATH not exist!"
    exit
fi

# CUDA_VISIBLE_DEVICES=1 python main.py --test_video --init_model=./model/final.pth --image_size=64 --motion_range=2 --num_inputs=3 --num_channel=3 --input_video_path=./video --output_flow_path=./flow 2>&1 | tee $MODEL_PATH/test_video.log

python visualize_flow_video.py --motion_range=2 --num_inputs=3 --input_video_path=./video --output_flow_path=./flow --output_flow_video_path=./flow_video  --output_flow_video_fps=10

cp test_video.sh $MODEL_PATH
