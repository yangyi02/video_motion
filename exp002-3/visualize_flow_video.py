import os
import sys
import numpy
import cv2
from PIL import Image
import matplotlib.animation as animation
from learning_args import parse_args
import flowlib
import logging
logging.basicConfig(format='[%(levelname)s %(asctime)s %(filename)s:%(lineno)s] %(message)s',
                            level=logging.INFO)


def get_file_list(args):
    video_folder, flow_folder = args.input_video_path, args.output_flow_path
    video_list, flow_list = {}, {}
    sub_dirs = os.listdir(flow_folder)
    for sub_dir in sub_dirs:
        video_files = os.listdir(os.path.join(video_folder, sub_dir))
        video_files.sort()
        video_files = [os.path.join(video_folder, sub_dir, f) for f in video_files]
        video_list[sub_dir] = video_files
        flow_files = os.listdir(os.path.join(flow_folder, sub_dir))
        flow_files.sort()
        flow_files = [os.path.join(flow_folder, sub_dir, f) for f in flow_files]
        flow_list[sub_dir] = flow_files
    return video_list, flow_list


def create_video(args, video_list, flow_list):
    output_folder = args.output_flow_video_path
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    for key in video_list.keys():
        video_files = video_list[key]
        flow_files = flow_list[key]
        create_one_video(args, video_files, flow_files, os.path.join(output_folder, key + '.avi'))


def create_one_video(args, video_files, flow_files, flow_video_file):
    m_range = args.motion_range
    start_idx = args.num_inputs
    vid = None
    for i in range(len(video_files)):
        if i < start_idx:
            im = numpy.array(Image.open(video_files[i]))
            flow = numpy.zeros((im.shape[0], im.shape[1], 2))
        else:
            logging.info('%s, %s' % (video_files[i], flow_files[i-start_idx]))
            im = numpy.array(Image.open(video_files[i]))
            flow = flowlib.read_flow(flow_files[i-start_idx])

        im_width, im_height = im.shape[1], im.shape[0]
        width, height = get_img_size(1, 2, im_width, im_height)
        img = numpy.ones((height, width, 3))

        x1, y1, x2, y2 = get_img_coordinate(1, 1, im_width, im_height)
        img[y1:y2, x1:x2, :] = im / 255.0

        optical_flow = flowlib.visualize_flow(flow, m_range)
        x1, y1, x2, y2 = get_img_coordinate(1, 2, im_width, im_height)
        img[y1:y2, x1:x2, :] = optical_flow / 255.0

        if vid is None:
            fps = 1
            vid = cv2.VideoWriter(flow_video_file, cv2.VideoWriter_fourcc(*'MJPG'), fps, (width, height))
        img = img * 255.0
        img = img.astype(numpy.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        vid.write(img)
    vid.release()


def get_img_size(n_row, n_col, im_width, im_height):
    height = n_row * im_height + (n_row - 1) * int(im_height/10)
    width = n_col * im_width + (n_col - 1) * int(im_width/10)
    return width, height


def get_img_coordinate(row, col, im_width, im_height):
    y1 = (row - 1) * im_height + (row - 1) * int(im_height/10)
    y2 = y1 + im_height
    x1 = (col - 1) * im_width + (col - 1) * int(im_width/10)
    x2 = x1 + im_width
    return x1, y1, x2, y2


def main():
    args = parse_args()
    logging.info(args)
    video_list, flow_list = get_file_list(args)
    create_video(args, video_list, flow_list)

if __name__ == '__main__':
    main()

