import argparse
import logging
logging.basicConfig(format='[%(levelname)s %(asctime)s %(filename)s:%(lineno)s] %(message)s',
                    level=logging.INFO)


def parse_args():
    arg_parser = argparse.ArgumentParser(description='video data organization', add_help=False)
    arg_parser.add_argument('--image_size', type=int, default=240)
    arg_parser.add_argument('--input_dir', default='/home/yi/Downloads/mpii-240')
    arg_parser.add_argument('--meta_file', default='./mpii-240/meta.pkl')
    arg_parser.add_argument('--second_meta_file', default='./mpii-240/meta_2.pkl')
    arg_parser.add_argument('--image_diff_file', default='./mpii-240/diff.txt')
    arg_parser.add_argument('--discontinue_file', default='./mpii-240/discontinue.txt')
    arg_parser.add_argument('--split_dir', default='/home/yi/Downloads/mpii-240-1')
    arg_parser.add_argument('--split_round', type=int, default=1)
    args = arg_parser.parse_args()
    return args


def main():
    args = parse_args()

if __name__ == '__main__':
    main()

