import os
import numpy
import pickle
import get_argument


def get_meta(args):
    input_dir = args.input_dir
    output_file = args.meta_file
    meta = {}
    cnt = 0
    image_dirs = os.listdir(input_dir)
    for image_dir in image_dirs:
        sub_dirs = os.listdir(os.path.join(input_dir, image_dir))
        sub_dirs.sort()
        for sub_dir in sub_dirs:
            files = os.listdir(os.path.join(input_dir, image_dir, sub_dir))
            files.sort()
            meta[cnt] = [image_dir, sub_dir, files]
            cnt += 1
    pickle.dump(meta, open(output_file, 'w'))


def main():
    args = get_argument.parse_args()
    get_meta(args)

if __name__ == '__main__':
    main()

