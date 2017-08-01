#!/bin/bash

python ../../visualize_flow_video.py --test_video --input_video_path=../video --output_flow_path=./flow --output_flow_video_path=./flow_video --motion_range=5 --num_inputs=4
