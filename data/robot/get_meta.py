import os
import numpy
import pickle


def get_meta():
    input_dir = '/home/yi/Downloads/robot-64'
    output_file = 'robot_64_meta.pkl'
    meta = {}
    cnt = 0
    for i in range(1, 1219):
        for j in range(25):
            files = ['%d.jpg' % k for k in range(25)]
            meta[cnt] = [str(i), str(j), files]
            cnt += 1
    pickle.dump(meta, open(output_file, 'w'))


def main():
    get_meta()

if __name__ == '__main__':
    main()

