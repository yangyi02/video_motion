import os
import numpy
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F

from learning_args import parse_args
from data import motion_dict, get_meta, generate_batch
from models import FullyConvNet, FullyConvResNet, UNet
from visualize import visualize
logging.basicConfig(format='[%(levelname)s %(asctime)s %(filename)s:%(lineno)s] %(message)s',
                            level=logging.INFO)


def train_unsupervised(args, model, m_kernel, reverse_m_dict):
    m_range = args.motion_range
    meta = get_meta(args, args.train_dir)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    base_loss = []
    train_loss = []
    best_test_loss = 1e10
    for epoch in range(args.train_epoch):
        optimizer.zero_grad()
        im_input, im_output = generate_batch(args, meta)
        im_input = Variable(torch.from_numpy(im_input).float())
        im_output = Variable(torch.from_numpy(im_output).float())
        if torch.cuda.is_available():
            im_input, im_output = im_input.cuda(), im_output.cuda()
        motion, disappear = model(im_input)
        m_mask = F.softmax(motion)
        im_last = im_input[:, -args.num_channel:, :, :]
        im_pred = construct_image(im_last, m_mask, disappear, m_kernel, m_range)
        loss = torch.abs(im_pred - im_output).sum()
        loss.backward()
        optimizer.step()
        train_loss.append(loss.data[0])
        base_loss.append(torch.abs(im_last - im_output).sum().data[0])
        if len(train_loss) > 1000:
            train_loss.pop(0)
        if len(base_loss) > 1000:
            base_loss.pop(0)
        ave_loss = sum(train_loss) / float(len(train_loss))
        ave_base_loss = sum(base_loss) / float(len(base_loss))
        logging.info('epoch %d, training loss: %.2f, average training loss: %.2f, base loss: %.2f', epoch, loss.data[0], ave_loss, ave_base_loss)
        if (epoch+1) % args.test_interval == 0:
            logging.info('epoch %d, testing', epoch)
            best_test_loss = validate(args, model, m_kernel, reverse_m_dict, best_test_loss)
    return model


def construct_image(im, m_mask, disappear, m_kernel, m_range):
    im = im * (1 - disappear).expand_as(im)
    pred = Variable(torch.Tensor(im.size()))
    if torch.cuda.is_available():
        pred = pred.cuda()
    for i in range(im.size(1)):
        im_expand = im[:, i, :, :].unsqueeze(1).expand_as(m_mask) * m_mask
        for j in range(im.size(0)):
            pred[j, i, :, :] = F.conv2d(im_expand[j, :, :, :].unsqueeze(0), m_kernel, None, 1, m_range)
    return pred


def validate(args, model, m_kernel, reverse_m_dict, best_test_loss):
    test_loss = test_unsupervised(args, model, m_kernel, reverse_m_dict)
    if test_loss <= best_test_loss:
        logging.info('model save to %s', os.path.join(args.save_dir, 'final.pth'))
        with open(os.path.join(args.save_dir, 'final.pth'), 'w') as handle:
            torch.save(model.state_dict(), handle)
        best_test_loss = test_loss
    logging.info('current best accuracy: %.2f', best_test_loss)
    return best_test_loss


def test_unsupervised(args, model, m_kernel, reverse_m_dict):
    m_range = args.motion_range
    meta = get_meta(args, args.test_dir)
    base_loss = []
    test_loss = []
    for epoch in range(args.test_epoch):
        im_input, im_output = generate_batch(args, meta)
        im_input = Variable(torch.from_numpy(im_input).float())
        im_output = Variable(torch.from_numpy(im_output).float())
        if torch.cuda.is_available():
            im_input, im_output = im_input.cuda(), im_output.cuda()
        motion, disappear = model(im_input)
        m_mask = F.softmax(motion)
        im_last = im_input[:, -args.num_channel:, :, :]
        im_pred = construct_image(im_last, m_mask, disappear, m_kernel, m_range)
        loss = torch.abs(im_pred - im_output).sum()
        test_loss.append(loss.data[0])
        base_loss.append(torch.abs(im_last - im_output).sum().data[0])
        if args.display:
            flow = motion2flow(m_mask, reverse_m_dict)
            visualize(im_input, im_output, im_pred, flow, disappear, m_range, reverse_m_dict)
    test_loss = numpy.mean(numpy.asarray(test_loss))
    base_loss = numpy.mean(numpy.asarray(base_loss))
    logging.info('average testing loss: %.2f, base loss: %.2f', test_loss, base_loss)
    return test_loss


def motion2flow(m_mask, reverse_m_dict):
    [batch_size, num_class, height, width] = m_mask.size()
    kernel_x = Variable(torch.zeros(batch_size, num_class, height, width))
    kernel_y = Variable(torch.zeros(batch_size, num_class, height, width))
    if torch.cuda.is_available():
        kernel_x = kernel_x.cuda()
        kernel_y = kernel_y.cuda()
    for i in range(num_class):
        (m_x, m_y) = reverse_m_dict[i]
        kernel_x[:, i, :, :] = m_x
        kernel_y[:, i, :, :] = m_y
    flow = Variable(torch.zeros(batch_size, 2, height, width))
    flow[:, 0, :, :] = (m_mask * kernel_x).sum(1)
    flow[:, 1, :, :] = (m_mask * kernel_y).sum(1)
    return flow


def main():
    args = parse_args()
    logging.info(args)
    m_dict, reverse_m_dict, m_kernel = motion_dict(args.motion_range)
    m_kernel = Variable(torch.from_numpy(m_kernel).float())
    height, width, channel = args.image_size, args.image_size, args.num_inputs * args.num_channel
    model = UNet(height, width, channel, len(m_dict))
    if torch.cuda.is_available():
        model = torch.nn.DataParallel(model).cuda()
        m_kernel = m_kernel.cuda()
    if args.train:
        model = train_unsupervised(args, model, m_kernel, reverse_m_dict)
    if args.test:
        model.load_state_dict(torch.load(args.init_model_path))
        test_unsupervised(args, model, m_kernel, reverse_m_dict)

if __name__ == '__main__':
    main()
