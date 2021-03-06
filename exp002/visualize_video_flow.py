import os
import sys
import numpy
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.colors as cl
import matplotlib.animation as animation
import logging
logging.basicConfig(format='[%(levelname)s %(asctime)s %(filename)s:%(lineno)s] %(message)s',
                            level=logging.INFO)


UNKNOWN_FLOW_THRESH = 1e7


def visualize(video_list, flow_list):
    for key in video_list.keys():
        video_files = video_list[key]
        flow_files = flow_list[key]
        visualize_video(video_files, flow_files)


def visualize_video(video_files, flow_files):
    for i in range(len(video_files)):
        if i == 0:
            im = numpy.array(Image.open(video_files[i]))
            flow = numpy.zeros((im.shape[0], im.shape[1], 2))
        else:
            im = numpy.array(Image.open(video_files[i]))
            flow = read_flow(flow_files[i-1])
        visualize_frame(im, flow)


def visualize_frame(im, flow):
    im_width, im_height = im.shape[1], im.shape[0]
    width, height = get_img_size(1, 2, im_width, im_height)
    img = numpy.ones((height, width, 3))

    x1, y1, x2, y2 = get_img_coordinate(1, 1, im_width, im_height)
    img[y1:y2, x1:x2, :] = im / 255.0

    optical_flow = visualize_flow(flow)
    # optical_flow = flow_to_image(flow)
    x1, y1, x2, y2 = get_img_coordinate(1, 2, im_width, im_height)
    img[y1:y2, x1:x2, :] = optical_flow / 255.0

    plt.figure(1)
    plt.imshow(img)
    plt.axis('off')
    plt.show()


def create_video(video_list, flow_list, start_idx=1, output_folder='flow_video'):
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    for key in video_list.keys():
        video_files = video_list[key]
        flow_files = flow_list[key]
        create_one_video(video_files, flow_files, os.path.join(output_folder, key + '.avi'), start_idx)


def create_one_video(video_files, flow_files, flow_video_file, start_idx):
    vid = None
    for i in range(len(video_files)):
        if i < start_idx:
            im = numpy.array(Image.open(video_files[i]))
            flow = numpy.zeros((im.shape[0], im.shape[1], 2))
        else:
            logging.info('%s, %s' % (video_files[i], flow_files[i-start_idx]))
            im = numpy.array(Image.open(video_files[i]))
            flow = read_flow(flow_files[i-start_idx])

        im_width, im_height = im.shape[1], im.shape[0]
        width, height = get_img_size(1, 2, im_width, im_height)
        img = numpy.ones((height, width, 3))

        x1, y1, x2, y2 = get_img_coordinate(1, 1, im_width, im_height)
        img[y1:y2, x1:x2, :] = im / 255.0

        optical_flow = visualize_flow(flow)
        # optical_flow = flow_to_image(flow)
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


def label2flow(motion_label, m_range, reverse_m_dict):
    motion = numpy.zeros((motion_label.shape[0], motion_label.shape[1], 2))
    for i in range(motion_label.shape[0]):
        for j in range(motion_label.shape[1]):
            motion[i, j, :] = numpy.asarray(reverse_m_dict[motion_label[i, j]])
    mag, ang = cv2.cartToPolar(motion[..., 0], motion[..., 1])
    hsv = numpy.zeros((motion.shape[0], motion.shape[1], 3), dtype=float)
    hsv[..., 0] = ang * 180 / numpy.pi / 2
    hsv[..., 1] = 255
    hsv[..., 2] = mag * 255.0 / m_range / numpy.sqrt(2)
    # hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
    rgb = cv2.cvtColor(hsv.astype(numpy.uint8), cv2.COLOR_HSV2BGR)
    return rgb


def visualize_flow(flow, mode='Y'):
    """
    this function visualize the input flow
    :param flow: input flow in array
    :param mode: choose which color mode to visualize the flow (Y: Ccbcr, RGB: RGB color)
    :return: None
    """
    if mode == 'Y':
        # Ccbcr color wheel
        img = flow_to_image(flow)
    elif mode == 'RGB':
        (h, w) = flow.shape[0:2]
        du = flow[:, :, 0]
        dv = flow[:, :, 1]
        valid = flow[:, :, 2]
        max_flow = max(numpy.max(du), numpy.max(dv))
        img = numpy.zeros((h, w, 3), dtype=numpy.float64)
        # angle layer
        img[:, :, 0] = numpy.arctan2(dv, du) / (2 * numpy.pi)
        # magnitude layer, normalized to 1
        img[:, :, 1] = numpy.sqrt(du * du + dv * dv) * 8 / max_flow
        # phase layer
        img[:, :, 2] = 8 - img[:, :, 1]
        # clip to [0,1]
        small_idx = img[:, :, 0:3] < 0
        large_idx = img[:, :, 0:3] > 1
        img[small_idx] = 0
        img[large_idx] = 1
        # convert to rgb
        img = cl.hsv_to_rgb(img)
        # remove invalid point
        img[:, :, 0] = img[:, :, 0] * valid
        img[:, :, 1] = img[:, :, 1] * valid
        img[:, :, 2] = img[:, :, 2] * valid

    return img


def flow_to_image(flow):
    """
    Convert flow into middlebury color code image
    :param flow: optical flow map
    :return: optical flow image in middlebury color
    """
    u = flow[:, :, 0]
    v = flow[:, :, 1]

    maxu = -999.
    maxv = -999.
    minu = 999.
    minv = 999.

    idxUnknow = (abs(u) > UNKNOWN_FLOW_THRESH) | (abs(v) > UNKNOWN_FLOW_THRESH)
    u[idxUnknow] = 0
    v[idxUnknow] = 0

    maxu = max(maxu, numpy.max(u))
    minu = min(minu, numpy.min(u))

    maxv = max(maxv, numpy.max(v))
    minv = min(minv, numpy.min(v))

    rad = numpy.sqrt(u ** 2 + v ** 2)
    maxrad = max(-1, numpy.max(rad))

    print "max flow: %.4f\nflow range:\nu = %.3f .. %.3f\nv = %.3f .. %.3f" % (maxrad, minu,maxu, minv, maxv)

    u = u/(maxrad + numpy.finfo(float).eps)
    v = v/(maxrad + numpy.finfo(float).eps)

    img = compute_color(u, v)

    idx = numpy.repeat(idxUnknow[:, :, numpy.newaxis], 3, axis=2)
    img[idx] = 0

    return numpy.uint8(img)


def compute_color(u, v):
    """
    compute optical flow color map
    :param u: optical flow horizontal map
    :param v: optical flow vertical map
    :return: optical flow in color code
    """
    [h, w] = u.shape
    img = numpy.zeros([h, w, 3])
    nanIdx = numpy.isnan(u) | numpy.isnan(v)
    u[nanIdx] = 0
    v[nanIdx] = 0

    colorwheel = make_color_wheel()
    ncols = numpy.size(colorwheel, 0)

    rad = numpy.sqrt(u**2+v**2)

    a = numpy.arctan2(-v, -u) / numpy.pi

    fk = (a+1) / 2 * (ncols - 1) + 1

    k0 = numpy.floor(fk).astype(int)

    k1 = k0 + 1
    k1[k1 == ncols+1] = 1
    f = fk - k0

    for i in range(0, numpy.size(colorwheel,1)):
        tmp = colorwheel[:, i]
        col0 = tmp[k0-1] / 255
        col1 = tmp[k1-1] / 255
        col = (1-f) * col0 + f * col1

        idx = rad <= 1
        col[idx] = 1-rad[idx]*(1-col[idx])
        notidx = numpy.logical_not(idx)

        col[notidx] *= 0.75
        img[:, :, i] = numpy.uint8(numpy.floor(255 * col*(1-nanIdx)))

    return img


def make_color_wheel():
    """
    Generate color wheel according Middlebury color code
    :return: Color wheel
    """
    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    ncols = RY + YG + GC + CB + BM + MR

    colorwheel = numpy.zeros([ncols, 3])

    col = 0

    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = numpy.transpose(numpy.floor(255*numpy.arange(0, RY) / RY))
    col += RY

    # YG
    colorwheel[col:col+YG, 0] = 255 - numpy.transpose(numpy.floor(255*numpy.arange(0, YG) / YG))
    colorwheel[col:col+YG, 1] = 255
    col += YG

    # GC
    colorwheel[col:col+GC, 1] = 255
    colorwheel[col:col+GC, 2] = numpy.transpose(numpy.floor(255*numpy.arange(0, GC) / GC))
    col += GC

    # CB
    colorwheel[col:col+CB, 1] = 255 - numpy.transpose(numpy.floor(255*numpy.arange(0, CB) / CB))
    colorwheel[col:col+CB, 2] = 255
    col += CB

    # BM
    colorwheel[col:col+BM, 2] = 255
    colorwheel[col:col+BM, 0] = numpy.transpose(numpy.floor(255*numpy.arange(0, BM) / BM))
    col += + BM

    # MR
    colorwheel[col:col+MR, 2] = 255 - numpy.transpose(numpy.floor(255 * numpy.arange(0, MR) / MR))
    colorwheel[col:col+MR, 0] = 255

    return colorwheel


def get_file_list():
    video_folder, flow_folder = 'video', 'flow'
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


def read_flow(filename):
    """
    read optical flow from Middlebury .flo file
    :param filename: name of the flow file
    :return: optical flow data in matrix
    """
    f = open(filename, 'rb')
    magic = numpy.fromfile(f, numpy.float32, count=1)
    data2d = None

    if 202021.25 != magic:
        print 'Magic number incorrect. Invalid .flo file'
    else:
        w = numpy.fromfile(f, numpy.int32, count=1)
        h = numpy.fromfile(f, numpy.int32, count=1)
        print "Reading %d x %d flo file" % (h, w)
        data2d = numpy.fromfile(f, numpy.float32, count=2 * w * h)
        # reshape data into 3D array (columns, rows, channels)
        data2d = numpy.resize(data2d, (h[0], w[0], 2))
    f.close()
    return data2d


def main():
    video_list, flow_list = get_file_list()
    # visualize(video_list, flow_list)
    start_idx = 4
    create_video(video_list, flow_list, start_idx)

if __name__ == '__main__':
    main()
