import os
import numpy
import pickle
import get_argument


def split_folder(args):
    meta_file = args.second_meta_file
    discontinue_file = args.discontinue_file
    split_dir = args.split_dir
    split_round = args.split_round
    meta = pickle.load(open(meta_file))
    idx_lines = open(discontinue_file).readlines()
    for line in idx_lines:
        items = line.strip().split('\t')
        image_dir = items[0]
        sub_dir = items[1]
        image_id = int(items[2])
        if image_id == 0:
            image_id = int(items[-1])
        files = os.listdir(os.path.join(split_dir, image_dir, sub_dir))
        files.sort()
        if not os.path.exists(os.path.join(split_dir, image_dir, sub_dir + '-' + str(split_round))):
            os.mkdir(os.path.join(split_dir, image_dir, sub_dir + '-' + str(split_round)))
        for i in range(image_id + 1, len(files)):
            file_name = os.path.join(split_dir, image_dir, sub_dir, files[i])
            new_file_name = os.path.join(split_dir, image_dir, sub_dir + '-' + str(split_round), files[i])
            cmd_str = 'mv %s %s' % (file_name, new_file_name)
            print(cmd_str)
            os.system(cmd_str)


def main():
    args = get_argument.parse_args()
    split_folder(args)

if __name__ == '__main__':
    main()

