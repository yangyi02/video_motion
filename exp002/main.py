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


def validate(args, model, meta, m_kernel, reverse_m_dict, best_test_loss):
    test_loss = test_unsupervised(args, model, meta, m_kernel, reverse_m_dict)
    if test_loss <= best_test_loss:
        logging.info('model save to %s', os.path.join(args.save_dir, 'final.pth'))
        with open(os.path.join(args.save_dir, 'final.pth'), 'w') as handle:
            torch.save(model.state_dict(), handle)
        best_test_loss = test_loss
    logging.info('current best accuracy: %.2f', best_test_loss)
    return best_test_loss


def test_unsupervised(args, model, meta, m_kernel, reverse_m_dict):
    m_range = args.motion_range
    test_loss = []
    for epoch in range(args.test_epoch):
        im_input, im_output = generate_batch(args, meta)
        im_input = Variable(torch.from_numpy(im_input).float())
        im_output = Variable(torch.from_numpy(im_output).float())
        if torch.cuda.is_available():
            im_input, im_output = im_input.cuda(), im_output.cuda()
        motion, disappear = model(im_input)
        im_input_last = im_input[:, -3:, :, :]
        im_pred = construct_image(im_input_last, motion, disappear, m_range, m_kernel)
        loss = torch.abs(im_pred - im_output).sum()
        if args.display:
            m_range = args.motion_range
            pred_motion = motion.max(1)[1]
            visualize(im_input_last, im_output, im_pred, pred_motion, disappear, m_range, reverse_m_dict)
        test_loss.append(loss.data[0])
    test_loss = numpy.mean(numpy.asarray(test_loss))
    logging.info('average testing loss: %.2f', test_loss)
    return test_loss


def construct_image(im, motion, disappear, m_range, m_kernel):
    appear_mask = 1 - disappear
    im = im * appear_mask.expand_as(im)
    mask = F.softmax(motion)
    pred = Variable(torch.Tensor(im.size(0), im.size(1), im.size(2), im.size(3)))
    if torch.cuda.is_available():
        pred = pred.cuda()
    for i in range(im.size(1)):
        im_expand = im[:, i, :, :].unsqueeze(1).expand_as(mask) * mask
        for j in range(im.size(0)):
            pred[j, i, :, :] = F.conv2d(im_expand[j, :, :, :].unsqueeze(0), m_kernel, None, 1, m_range)
    return pred


def train_unsupervised(args, model, meta, m_kernel, reverse_m_dict):
    m_range = args.motion_range
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
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
        im_input_last = im_input[:, -3:, :, :]
        im_pred = construct_image(im_input_last, motion, disappear, m_range, m_kernel)
        loss = torch.abs(im_pred - im_output).sum()
        loss.backward()
        optimizer.step()
        train_loss.append(loss.data[0])
        if len(train_loss) > 1000:
            train_loss.pop(0)
        ave_loss = sum(train_loss) / float(len(train_loss))
        logging.info('epoch %d, training loss: %.2f, average training loss: %.2f', epoch, loss.data[0], ave_loss)
        if (epoch+1) % args.test_interval == 0:
            logging.info('epoch %d, testing', epoch)
            best_test_loss = validate(args, model, meta, m_kernel, reverse_m_dict, best_test_loss)
    return model


def main():
    args = parse_args()
    logging.info(args)
    m_dict, reverse_m_dict, m_kernel = motion_dict(args.motion_range)
    m_kernel = Variable(torch.from_numpy(m_kernel).float())
    meta = get_meta(args)
    im_input, im_output = generate_batch(args, meta)
    [_, im_channel, args.image_height, args.image_width] = im_input.shape
    model = UNet(args.image_height, args.image_width, im_channel, len(m_dict))
    if torch.cuda.is_available():
        model = torch.nn.DataParallel(model).cuda()
        m_kernel = m_kernel.cuda()
    if args.train:
        model = train_unsupervised(args, model, meta, m_kernel, reverse_m_dict)
    if args.test:
        model.load_state_dict(torch.load(args.init_model_path))
        test_unsupervised(args, model, meta, m_kernel, reverse_m_dict)

if __name__ == '__main__':
    main()
