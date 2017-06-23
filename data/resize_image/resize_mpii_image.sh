#!/usr/bin/env bash

INPUT_DIR=/home/yi/Downloads/mpii_human_pose_v1_sequences
OUTPUT_DIR=/home/yi/Downloads/mpii-240
FILE_LIST=../mpii/mpii_file.list
SIZE=240

python create_file_list.py --input_dir=${INPUT_DIR} --output_file=${FILE_LIST}
# python resize_image.py --input_dir=${INPUT_DIR} --output_dir=${OUTPUT_DIR} --size=${SIZE} --file_list=${FILE_LIST}
python resize_image_mp.py --input_dir=${INPUT_DIR} --output_dir=${OUTPUT_DIR} --size=${SIZE} --file_list=${FILE_LIST}

