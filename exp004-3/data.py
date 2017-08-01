import os
import numpy
import matplotlib.pyplot as plt
from skimage import io, transform
import pickle
from PIL import Image

import learning_args
import logging
logging.basicConfig(format='[%(levelname)s %(asctime)s %(filename)s:%(lineno)s] %(message)s',
                            level=logging.INFO)


def motion_dict(m_range):
    m_dict, reverse_m_dict = {}, {}
    x = numpy.linspace(-m_range, m_range, 2 * m_range + 1)
    y = numpy.linspace(-m_range, m_range, 2 * m_range + 1)
    m_x, m_y = numpy.meshgrid(x, y)
    m_x, m_y, = m_x.reshape(-1).astype(int), m_y.reshape(-1).astype(int)
    m_kernel = numpy.zeros((1, len(m_x), 2 * m_range + 1, 2 * m_range + 1))
    for i in range(len(m_x)):
        m_dict[(m_x[i], m_y[i])] = i
        reverse_m_dict[i] = (m_x[i], m_y[i])
        m_kernel[:, i, m_y[i] + m_range, m_x[i] + m_range] = 1
    return m_dict, reverse_m_dict, m_kernel


def get_meta(args, image_dir):
    meta = {}
    cnt = 0
    sub_dirs = os.listdir(image_dir)
    for sub_dir in sub_dirs:
        sub_sub_dirs = os.listdir(os.path.join(image_dir, sub_dir))
        for sub_sub_dir in sub_sub_dirs:
            image_files = os.listdir(os.path.join(image_dir, sub_dir, sub_sub_dir))
            image_files.sort(key=lambda f: int(filter(str.isdigit, f)))
            image_names = [os.path.join(image_dir, sub_dir, sub_sub_dir, f) for f in image_files]
            num_images = len(image_names)
            if num_images < 2 * args.num_inputs + 1:
                continue
            idx = range(0, num_images - 2 * args.num_inputs)
            for i in range(len(idx)):
                start_idx = idx[i]
                end_idx = idx[i] + 2 * args.num_inputs + 1
                meta[cnt] = image_names[start_idx:end_idx]
                cnt += 1
    return meta


def generate_batch(args, meta):
    batch_size, n_inputs, height, width, im_channel = args.batch_size, args.num_inputs, args.image_size, args.image_size, args.num_channel
    idx = numpy.random.permutation(len(meta))
    images = numpy.zeros((batch_size, 2*n_inputs+1, im_channel, height, width))
    i = 0
    cnt = 0
    while i < batch_size:
        image_names = meta[idx[cnt]]
        n_image = len(image_names)
        assert n_image == 2 * n_inputs + 1
        for j in range(n_image):
            image_name = image_names[j]
            if args.num_channel == 1:
                im = numpy.array(Image.open(image_name).convert('L')) / 255.0
            else:
                im = numpy.array(Image.open(image_name)) / 255.0
            if args.num_channel == 1:
                im = numpy.expand_dims(im, 0)
            else:
                im = im.transpose((2, 0, 1))
            if j == 0:
                _, im_height, im_width = im.shape
                idx_h = numpy.random.randint(0, im_height+1-height)
                idx_w = numpy.random.randint(0, im_width+1-width)
            images[i, j, :, :, :] = im[:, idx_h:idx_h+height, idx_w:idx_w+width]
        im_diff = []
        for j in range(2 * n_inputs):
            diff = images[i, j, :, :, :] - images[i, j+1, :, :, :]
            diff = numpy.abs(diff)
            diff = numpy.sum(diff) / args.num_channel / height / width
            im_diff.append(diff)
        cnt = cnt + 1
        im_diff = numpy.asarray(im_diff)
        if any(im_diff < 0.02) or any(im_diff > 0.1):
            continue
        im_diff_div = im_diff / (numpy.median(im_diff) + 1e-5)
        # if any(numpy.log(im_diff_div) > 0.1) or any(numpy.log(im_diff_div) < -0.1):
        if any(im_diff_div > 1.25) or any(im_diff_div < 0.8):
            continue
        # print im_diff
        i = i + 1
    im_input_f = images[:, :n_inputs, :, :, :]
    im_input_f = im_input_f.reshape(batch_size, n_inputs*im_channel, height, width)
    im_input_b = images[:, -n_inputs:, :, :, :]
    im_input_b = im_input_b[:, ::-1, :, :, :]
    im_input_b = im_input_b.reshape(batch_size, n_inputs*im_channel, height, width)
    im_output = images[:, n_inputs, :, :, :]
    im_output = im_output.squeeze()
    return im_input_f, im_input_b, im_output


def display(images1, images2, images3, images4, images5):
    for i in range(images1.shape[0]):
        plt.figure(1)
        plt.subplot(2, 5, 1)
        if images1.shape[1] == 1:
            im1 = images1[i, :, :, :].squeeze()
            plt.imshow(im1, cmap='gray')
        else:
            im1 = images1[i, :, :, :].squeeze().transpose(1, 2, 0)
            plt.imshow(im1)
        plt.subplot(2, 5, 2)
        if images2.shape[1] == 1:
            im2 = images2[i, :, :, :].squeeze()
            plt.imshow(im2, cmap='gray')
        else:
            im2 = images2[i, :, :, :].squeeze().transpose(1, 2, 0)
            plt.imshow(im2)
        plt.subplot(2, 5, 3)
        if images3.shape[1] == 1:
            im3 = images3[i, :, :, :].squeeze()
            plt.imshow(im3, cmap='gray')
        else:
            im3 = images3[i, :, :, :].squeeze().transpose(1, 2, 0)
            plt.imshow(im3)
        plt.subplot(2, 5, 4)
        if images4.shape[1] == 1:
            im4 = images4[i, :, :, :].squeeze()
            plt.imshow(im4, cmap='gray')
        else:
            im4 = images4[i, :, :, :].squeeze().transpose(1, 2, 0)
            plt.imshow(im4)
        plt.subplot(2, 5, 5)
        if images5.shape[1] == 1:
            im5 = images5[i, :, :, :].squeeze()
            plt.imshow(im5, cmap='gray')
        else:
            im5 = images5[i, :, :, :].squeeze().transpose(1, 2, 0)
            plt.imshow(im5)
        plt.subplot(2, 5, 6)
        im_diff1 = abs(im2 - im1)
        plt.imshow(im_diff1)
        plt.subplot(2, 5, 7)
        im_diff2 = abs(im3 - im2)
        plt.imshow(im_diff2)
        plt.subplot(2, 5, 9)
        im_diff3 = abs(im4 - im3)
        plt.imshow(im_diff3)
        plt.subplot(2, 5, 10)
        im_diff4 = abs(im5 - im4)
        plt.imshow(im_diff4)
        plt.show()


def unit_test():
    m_dict, reverse_m_dict, m_kernel = motion_dict(1)
    args = learning_args.parse_args()
    meta = get_meta(args, args.train_dir)
    print len(meta)
    im_input_f, im_input_b, im_output = generate_batch(args, meta)
    if True:
        im1 = im_input_f[:, -args.num_channel*2:-args.num_channel, :, :]
        im2 = im_input_f[:, -args.num_channel:, :, :]
        im3 = im_output
        im4 = im_input_b[:, -args.num_channel:, :, :]
        im5 = im_input_b[:, -args.num_channel*2:-args.num_channel, :, :]
        print im1.shape, im2.shape, im3.shape, im4.shape, im5.shape
        display(im1, im2, im3, im4, im5)

if __name__ == '__main__':
    unit_test()

