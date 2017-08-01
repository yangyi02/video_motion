import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F


class FullyConvNet(nn.Module):
    def __init__(self, im_size, im_channel, n_class):
        super(FullyConvNet, self).__init__()
        num_hidden = 32
        self.conv0 = nn.Conv2d(im_channel, num_hidden, 3, 1, 1)
        self.bn0 = nn.BatchNorm2d(num_hidden)
        self.conv1 = nn.Conv2d(num_hidden, num_hidden, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(num_hidden)
        self.conv2 = nn.Conv2d(num_hidden, num_hidden, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(num_hidden)
        self.conv3 = nn.Conv2d(num_hidden, num_hidden, 3, 1, 1)
        self.bn3 = nn.BatchNorm2d(num_hidden)
        self.conv4 = nn.Conv2d(num_hidden, num_hidden, 3, 1, 1)
        self.bn4 = nn.BatchNorm2d(num_hidden)
        self.conv5 = nn.Conv2d(num_hidden, num_hidden, 3, 1, 1)
        self.bn5 = nn.BatchNorm2d(num_hidden)
        self.conv6 = nn.Conv2d(num_hidden, num_hidden, 3, 1, 1)
        self.bn6 = nn.BatchNorm2d(num_hidden)
        self.conv = nn.Conv2d(num_hidden, n_class, 3, 1, 1)
        self.im_size = im_size
        self.im_channel = im_channel
        self.n_class = n_class

    def forward(self, x):
        x = self.bn0(self.conv0(x))
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        motion = self.conv(x)
        return motion


class FullyConvResNet(nn.Module):
    def __init__(self, im_size, im_channel, n_class):
        super(FullyConvResNet, self).__init__()
        num_hidden = 32
        self.conv0 = nn.Conv2d(2*im_channel, num_hidden, 3, 1, 1)
        self.bn0 = nn.BatchNorm2d(num_hidden)
        self.conv1_1 = nn.Conv2d(num_hidden, num_hidden, 3, 1, 1)
        self.bn1_1 = nn.BatchNorm2d(num_hidden)
        self.conv1_2 = nn.Conv2d(num_hidden, num_hidden, 3, 1, 1)
        self.bn1_2 = nn.BatchNorm2d(num_hidden)
        self.conv2_1 = nn.Conv2d(num_hidden, num_hidden, 3, 1, 1)
        self.bn2_1 = nn.BatchNorm2d(num_hidden)
        self.conv2_2 = nn.Conv2d(num_hidden, num_hidden, 3, 1, 1)
        self.bn2_2 = nn.BatchNorm2d(num_hidden)
        self.conv3_1 = nn.Conv2d(num_hidden, num_hidden, 3, 1, 1)
        self.bn3_1 = nn.BatchNorm2d(num_hidden)
        self.conv3_2 = nn.Conv2d(num_hidden, num_hidden, 3, 1, 1)
        self.bn3_2 = nn.BatchNorm2d(num_hidden)
        self.conv4_1 = nn.Conv2d(num_hidden, num_hidden, 3, 1, 1)
        self.bn4_1 = nn.BatchNorm2d(num_hidden)
        self.conv4_2 = nn.Conv2d(num_hidden, num_hidden, 3, 1, 1)
        self.bn4_2 = nn.BatchNorm2d(num_hidden)
        self.conv5_1 = nn.Conv2d(num_hidden, num_hidden, 3, 1, 1)
        self.bn5_1 = nn.BatchNorm2d(num_hidden)
        self.conv5_2 = nn.Conv2d(num_hidden, num_hidden, 3, 1, 1)
        self.bn5_2 = nn.BatchNorm2d(num_hidden)
        self.conv6_1 = nn.Conv2d(num_hidden, num_hidden, 3, 1, 1)
        self.bn6_1 = nn.BatchNorm2d(num_hidden)
        self.conv6_2 = nn.Conv2d(num_hidden, num_hidden, 3, 1, 1)
        self.bn6_2 = nn.BatchNorm2d(num_hidden)
        self.conv = nn.Conv2d(num_hidden, n_class, 3, 1, 1)
        self.im_size = im_size
        self.im_channel = im_channel
        self.n_class = n_class

    def forward(self, x1, x2):
        x = torch.cat((x1, x2), 1)
        x = self.bn0(self.conv0(x))
        y = F.relu(self.bn1_1(self.conv1_1(x)))
        x = F.relu(self.bn1_2(self.conv1_2(y) + x))
        y = F.relu(self.bn2_1(self.conv2_1(x)))
        x = F.relu(self.bn2_2(self.conv2_2(y) + x))
        y = F.relu(self.bn3_1(self.conv3_1(x)))
        x = F.relu(self.bn3_2(self.conv3_2(y) + x))
        y = F.relu(self.bn4_1(self.conv4_1(x)))
        x = F.relu(self.bn4_2(self.conv4_2(y) + x))
        y = F.relu(self.bn5_1(self.conv5_1(x)))
        x = F.relu(self.bn5_2(self.conv5_2(y) + x))
        y = F.relu(self.bn6_1(self.conv6_1(x)))
        x = F.relu(self.bn6_2(self.conv6_2(y) + x))
        motion = self.conv(x)
        return motion


class UNet(nn.Module):
    def __init__(self, im_height, im_width, im_channel, n_class):
        super(UNet, self).__init__()
        num_hidden = 64
        self.conv0 = nn.Conv2d(im_channel, num_hidden, 3, 1, 1)
        self.bn0 = nn.BatchNorm2d(num_hidden)
        self.conv1 = nn.Conv2d(num_hidden, num_hidden, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(num_hidden)
        self.conv2 = nn.Conv2d(num_hidden, num_hidden, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(num_hidden)
        self.conv3 = nn.Conv2d(num_hidden, num_hidden, 3, 1, 1)
        self.bn3 = nn.BatchNorm2d(num_hidden)
        self.conv4 = nn.Conv2d(num_hidden, num_hidden, 3, 1, 1)
        self.bn4 = nn.BatchNorm2d(num_hidden)
        self.conv5 = nn.Conv2d(num_hidden, num_hidden, 3, 1, 1)
        self.bn5 = nn.BatchNorm2d(num_hidden)
        self.conv6 = nn.Conv2d(num_hidden, num_hidden, 3, 1, 1)
        self.bn6 = nn.BatchNorm2d(num_hidden)
        self.conv7 = nn.Conv2d(num_hidden, num_hidden, 3, 1, 1)
        self.bn7 = nn.BatchNorm2d(num_hidden)
        self.conv8 = nn.Conv2d(num_hidden*2, num_hidden, 3, 1, 1)
        self.bn8 = nn.BatchNorm2d(num_hidden)
        self.conv9 = nn.Conv2d(num_hidden*2, num_hidden, 3, 1, 1)
        self.bn9 = nn.BatchNorm2d(num_hidden)
        self.conv10 = nn.Conv2d(num_hidden*2, num_hidden, 3, 1, 1)
        self.bn10 = nn.BatchNorm2d(num_hidden)
        self.conv11 = nn.Conv2d(num_hidden*2, num_hidden, 3, 1, 1)
        self.bn11 = nn.BatchNorm2d(num_hidden)
        self.conv12 = nn.Conv2d(num_hidden*2, num_hidden, 3, 1, 1)
        self.bn12 = nn.BatchNorm2d(num_hidden)
        self.maxpool = nn.MaxPool2d(2, stride=2, return_indices=False, ceil_mode=False)
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)
        self.conv = nn.Conv2d(num_hidden*2, n_class, 3, 1, 1)
        self.im_height = im_height
        self.im_width = im_width
        self.im_channel = im_channel
        self.n_class = n_class

    def forward(self, x):
        x = self.bn0(self.conv0(x))
        x1 = F.relu(self.bn1(self.conv1(x)))
        x2 = self.maxpool(x1)
        x2 = F.relu(self.bn2(self.conv2(x2)))
        x3 = self.maxpool(x2)
        x3 = F.relu(self.bn3(self.conv3(x3)))
        x4 = self.maxpool(x3)
        x4 = F.relu(self.bn4(self.conv4(x4)))
        x5 = self.maxpool(x4)
        x5 = F.relu(self.bn5(self.conv5(x5)))
        x6 = self.maxpool(x5)
        x6 = F.relu(self.bn6(self.conv6(x6)))
        x7 = self.maxpool(x6)
        x7 = F.relu(self.bn7(self.conv7(x7)))
        x7 = self.upsample(x7)
        x8 = torch.cat((x7, x6), 1)
        x8 = F.relu(self.bn8(self.conv8(x8)))
        x8 = self.upsample(x8)
        x9 = torch.cat((x8, x5), 1)
        x9 = F.relu(self.bn9(self.conv9(x9)))
        x9 = self.upsample(x9)
        x10 = torch.cat((x9, x4), 1)
        x10 = F.relu(self.bn10(self.conv10(x10)))
        x10 = self.upsample(x10)
        x11 = torch.cat((x10, x3), 1)
        x11 = F.relu(self.bn11(self.conv11(x11)))
        x11 = self.upsample(x11)
        x12 = torch.cat((x11, x2), 1)
        x12 = F.relu(self.bn12(self.conv12(x12)))
        x12 = self.upsample(x12)
        x = torch.cat((x12, x1), 1)
        motion = self.conv(x)
        return motion


class UNetBidirection(nn.Module):
    def __init__(self, im_height, im_width, im_channel, n_inputs, n_class, m_range, m_kernel):
        super(UNetBidirection, self).__init__()
        num_hidden = 64
        self.conv0 = nn.Conv2d(im_channel * n_inputs, num_hidden, 3, 1, 1)
        self.bn0 = nn.BatchNorm2d(num_hidden)
        self.conv1 = nn.Conv2d(num_hidden, num_hidden, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(num_hidden)
        self.conv2 = nn.Conv2d(num_hidden, num_hidden, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(num_hidden)
        self.conv3 = nn.Conv2d(num_hidden, num_hidden, 3, 1, 1)
        self.bn3 = nn.BatchNorm2d(num_hidden)
        self.conv4 = nn.Conv2d(num_hidden, num_hidden, 3, 1, 1)
        self.bn4 = nn.BatchNorm2d(num_hidden)
        self.conv5 = nn.Conv2d(num_hidden, num_hidden, 3, 1, 1)
        self.bn5 = nn.BatchNorm2d(num_hidden)
        self.conv6 = nn.Conv2d(num_hidden, num_hidden, 3, 1, 1)
        self.bn6 = nn.BatchNorm2d(num_hidden)
        self.conv7 = nn.Conv2d(num_hidden, num_hidden, 3, 1, 1)
        self.bn7 = nn.BatchNorm2d(num_hidden)
        self.conv8 = nn.Conv2d(num_hidden*2, num_hidden, 3, 1, 1)
        self.bn8 = nn.BatchNorm2d(num_hidden)
        self.conv9 = nn.Conv2d(num_hidden*2, num_hidden, 3, 1, 1)
        self.bn9 = nn.BatchNorm2d(num_hidden)
        self.conv10 = nn.Conv2d(num_hidden*2, num_hidden, 3, 1, 1)
        self.bn10 = nn.BatchNorm2d(num_hidden)
        self.conv11 = nn.Conv2d(num_hidden*2, num_hidden, 3, 1, 1)
        self.bn11 = nn.BatchNorm2d(num_hidden)
        self.conv12 = nn.Conv2d(num_hidden*2, num_hidden, 3, 1, 1)
        self.bn12 = nn.BatchNorm2d(num_hidden)
        self.maxpool = nn.MaxPool2d(2, stride=2, return_indices=False, ceil_mode=False)
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)
        self.conv = nn.Conv2d(num_hidden*2, n_class, 3, 1, 1)
        self.conv_a = nn.Conv2d(2, 1, 1, 1, 0)
        self.im_height = im_height
        self.im_width = im_width
        self.im_channel = im_channel
        self.n_class = n_class
        self.m_range = m_range
        self.m_kernel = m_kernel

    def forward(self, im_input_f, im_input_b, ones):
        x = self.bn0(self.conv0(im_input_f))
        x1 = F.relu(self.bn1(self.conv1(x)))
        x2 = self.maxpool(x1)
        x2 = F.relu(self.bn2(self.conv2(x2)))
        x3 = self.maxpool(x2)
        x3 = F.relu(self.bn3(self.conv3(x3)))
        x4 = self.maxpool(x3)
        x4 = F.relu(self.bn4(self.conv4(x4)))
        x5 = self.maxpool(x4)
        x5 = F.relu(self.bn5(self.conv5(x5)))
        x6 = self.maxpool(x5)
        x6 = F.relu(self.bn6(self.conv6(x6)))
        x7 = self.maxpool(x6)
        x7 = F.relu(self.bn7(self.conv7(x7)))
        x7 = self.upsample(x7)
        x8 = torch.cat((x7, x6), 1)
        x8 = F.relu(self.bn8(self.conv8(x8)))
        x8 = self.upsample(x8)
        x9 = torch.cat((x8, x5), 1)
        x9 = F.relu(self.bn9(self.conv9(x9)))
        x9 = self.upsample(x9)
        x10 = torch.cat((x9, x4), 1)
        x10 = F.relu(self.bn10(self.conv10(x10)))
        x10 = self.upsample(x10)
        x11 = torch.cat((x10, x3), 1)
        x11 = F.relu(self.bn11(self.conv11(x11)))
        x11 = self.upsample(x11)
        x12 = torch.cat((x11, x2), 1)
        x12 = F.relu(self.bn12(self.conv12(x12)))
        x12 = self.upsample(x12)
        x = torch.cat((x12, x1), 1)
        motion_f = self.conv(x)

        x = self.bn0(self.conv0(im_input_b))
        x1 = F.relu(self.bn1(self.conv1(x)))
        x2 = self.maxpool(x1)
        x2 = F.relu(self.bn2(self.conv2(x2)))
        x3 = self.maxpool(x2)
        x3 = F.relu(self.bn3(self.conv3(x3)))
        x4 = self.maxpool(x3)
        x4 = F.relu(self.bn4(self.conv4(x4)))
        x5 = self.maxpool(x4)
        x5 = F.relu(self.bn5(self.conv5(x5)))
        x6 = self.maxpool(x5)
        x6 = F.relu(self.bn6(self.conv6(x6)))
        x7 = self.maxpool(x6)
        x7 = F.relu(self.bn7(self.conv7(x7)))
        x7 = self.upsample(x7)
        x8 = torch.cat((x7, x6), 1)
        x8 = F.relu(self.bn8(self.conv8(x8)))
        x8 = self.upsample(x8)
        x9 = torch.cat((x8, x5), 1)
        x9 = F.relu(self.bn9(self.conv9(x9)))
        x9 = self.upsample(x9)
        x10 = torch.cat((x9, x4), 1)
        x10 = F.relu(self.bn10(self.conv10(x10)))
        x10 = self.upsample(x10)
        x11 = torch.cat((x10, x3), 1)
        x11 = F.relu(self.bn11(self.conv11(x11)))
        x11 = self.upsample(x11)
        x12 = torch.cat((x11, x2), 1)
        x12 = F.relu(self.bn12(self.conv12(x12)))
        x12 = self.upsample(x12)
        x = torch.cat((x12, x1), 1)
        motion_b = self.conv(x)

        m_mask_f = F.softmax(motion_f)
        m_mask_b = F.softmax(motion_b)

        seg_f = construct_seg(ones, m_mask_f, self.m_kernel, self.m_range)
        seg_b = construct_seg(ones, m_mask_b, self.m_kernel, self.m_range)

        pred_f = construct_image(im_input_f[:, -self.im_channel:, :, :], m_mask_f, self.m_kernel, self.m_range)
        pred_b = construct_image(im_input_b[:, -self.im_channel:, :, :], m_mask_b, self.m_kernel, self.m_range)

        seg = torch.cat((seg_f, seg_b), 1)
        attn = self.conv_a(seg)
        attn = F.sigmoid(attn)
        pred = attn.expand_as(pred_f) * pred_f + (1 - attn.expand_as(pred_b)) * pred_b
        return pred, pred_f, m_mask_f, attn, pred_b, m_mask_b, 1 - attn


def construct_seg(ones, m_mask, m_kernel, m_range):
    ones_expand = ones.expand_as(m_mask) * m_mask
    seg = Variable(torch.Tensor(ones.size()))
    if torch.cuda.is_available():
        seg = seg.cuda()
    for i in range(ones.size(0)):
        seg[i, :, :, :] = F.conv2d(ones_expand[i, :, :, :].unsqueeze(0), m_kernel, None, 1, m_range)
    return seg


def construct_image(im, m_mask, m_kernel, m_range):
    pred = Variable(torch.Tensor(im.size()))
    if torch.cuda.is_available():
        pred = pred.cuda()
    for i in range(im.size(1)):
        im_expand = im[:, i, :, :].unsqueeze(1).expand_as(m_mask) * m_mask
        for j in range(im.size(0)):
            pred[j, i, :, :] = F.conv2d(im_expand[j, :, :, :].unsqueeze(0), m_kernel, None, 1, m_range)
    return pred


class UNetBidirection2(nn.Module):
    def __init__(self, im_height, im_width, im_channel, n_inputs, n_class, m_range, m_kernel):
        super(UNetBidirection2, self).__init__()
        num_hidden = 64
        self.conv0 = nn.Conv2d(im_channel * n_inputs, num_hidden, 3, 1, 1)
        self.bn0 = nn.BatchNorm2d(num_hidden)
        self.conv1 = nn.Conv2d(num_hidden, num_hidden, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(num_hidden)
        self.conv2 = nn.Conv2d(num_hidden, num_hidden, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(num_hidden)
        self.conv3 = nn.Conv2d(num_hidden, num_hidden, 3, 1, 1)
        self.bn3 = nn.BatchNorm2d(num_hidden)
        self.conv4 = nn.Conv2d(num_hidden, num_hidden, 3, 1, 1)
        self.bn4 = nn.BatchNorm2d(num_hidden)
        self.conv5 = nn.Conv2d(num_hidden, num_hidden, 3, 1, 1)
        self.bn5 = nn.BatchNorm2d(num_hidden)
        self.conv6 = nn.Conv2d(num_hidden, num_hidden, 3, 1, 1)
        self.bn6 = nn.BatchNorm2d(num_hidden)
        self.conv7 = nn.Conv2d(num_hidden, num_hidden, 3, 1, 1)
        self.bn7 = nn.BatchNorm2d(num_hidden)
        self.conv8 = nn.Conv2d(num_hidden*2, num_hidden, 3, 1, 1)
        self.bn8 = nn.BatchNorm2d(num_hidden)
        self.conv9 = nn.Conv2d(num_hidden*2, num_hidden, 3, 1, 1)
        self.bn9 = nn.BatchNorm2d(num_hidden)
        self.conv10 = nn.Conv2d(num_hidden*2, num_hidden, 3, 1, 1)
        self.bn10 = nn.BatchNorm2d(num_hidden)
        self.conv11 = nn.Conv2d(num_hidden*2, num_hidden, 3, 1, 1)
        self.bn11 = nn.BatchNorm2d(num_hidden)
        self.conv12 = nn.Conv2d(num_hidden*2, num_hidden, 3, 1, 1)
        self.bn12 = nn.BatchNorm2d(num_hidden)
        self.maxpool = nn.MaxPool2d(2, stride=2, return_indices=False, ceil_mode=False)
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)
        self.conv = nn.Conv2d(num_hidden*2, n_class, 3, 1, 1)
        self.conv_a = nn.Conv2d(2, 1, 1, 1, 0)
        self.im_height = im_height
        self.im_width = im_width
        self.im_channel = im_channel
        self.n_class = n_class
        self.m_range = m_range
        self.m_kernel = m_kernel

    def forward(self, im_input_f, im_input_b, ones):
        x = self.bn0(self.conv0(im_input_f))
        x1 = F.relu(self.bn1(self.conv1(x)))
        x2 = self.maxpool(x1)
        x2 = F.relu(self.bn2(self.conv2(x2)))
        x3 = self.maxpool(x2)
        x3 = F.relu(self.bn3(self.conv3(x3)))
        x4 = self.maxpool(x3)
        x4 = F.relu(self.bn4(self.conv4(x4)))
        x5 = self.maxpool(x4)
        x5 = F.relu(self.bn5(self.conv5(x5)))
        x6 = self.maxpool(x5)
        x6 = F.relu(self.bn6(self.conv6(x6)))
        x7 = self.maxpool(x6)
        x7 = F.relu(self.bn7(self.conv7(x7)))
        x7 = self.upsample(x7)
        x8 = torch.cat((x7, x6), 1)
        x8 = F.relu(self.bn8(self.conv8(x8)))
        x8 = self.upsample(x8)
        x9 = torch.cat((x8, x5), 1)
        x9 = F.relu(self.bn9(self.conv9(x9)))
        x9 = self.upsample(x9)
        x10 = torch.cat((x9, x4), 1)
        x10 = F.relu(self.bn10(self.conv10(x10)))
        x10 = self.upsample(x10)
        x11 = torch.cat((x10, x3), 1)
        x11 = F.relu(self.bn11(self.conv11(x11)))
        x11 = self.upsample(x11)
        x12 = torch.cat((x11, x2), 1)
        x12 = F.relu(self.bn12(self.conv12(x12)))
        x12 = self.upsample(x12)
        x = torch.cat((x12, x1), 1)
        motion_f = self.conv(x)

        x = self.bn0(self.conv0(im_input_b))
        x1 = F.relu(self.bn1(self.conv1(x)))
        x2 = self.maxpool(x1)
        x2 = F.relu(self.bn2(self.conv2(x2)))
        x3 = self.maxpool(x2)
        x3 = F.relu(self.bn3(self.conv3(x3)))
        x4 = self.maxpool(x3)
        x4 = F.relu(self.bn4(self.conv4(x4)))
        x5 = self.maxpool(x4)
        x5 = F.relu(self.bn5(self.conv5(x5)))
        x6 = self.maxpool(x5)
        x6 = F.relu(self.bn6(self.conv6(x6)))
        x7 = self.maxpool(x6)
        x7 = F.relu(self.bn7(self.conv7(x7)))
        x7 = self.upsample(x7)
        x8 = torch.cat((x7, x6), 1)
        x8 = F.relu(self.bn8(self.conv8(x8)))
        x8 = self.upsample(x8)
        x9 = torch.cat((x8, x5), 1)
        x9 = F.relu(self.bn9(self.conv9(x9)))
        x9 = self.upsample(x9)
        x10 = torch.cat((x9, x4), 1)
        x10 = F.relu(self.bn10(self.conv10(x10)))
        x10 = self.upsample(x10)
        x11 = torch.cat((x10, x3), 1)
        x11 = F.relu(self.bn11(self.conv11(x11)))
        x11 = self.upsample(x11)
        x12 = torch.cat((x11, x2), 1)
        x12 = F.relu(self.bn12(self.conv12(x12)))
        x12 = self.upsample(x12)
        x = torch.cat((x12, x1), 1)
        motion_b = self.conv(x)

        m_mask_f = F.softmax(motion_f)
        m_mask_b = F.softmax(motion_b)

        seg_f = construct_seg(ones, m_mask_f, self.m_kernel, self.m_range)
        seg_b = construct_seg(ones, m_mask_b, self.m_kernel, self.m_range)

        pred_f = construct_image(im_input_f[:, -self.im_channel:, :, :], m_mask_f, self.m_kernel, self.m_range)
        pred_b = construct_image(im_input_b[:, -self.im_channel:, :, :], m_mask_b, self.m_kernel, self.m_range)

        attn = (seg_f + 1e-5) / (seg_f + seg_b + 2e-5)
        pred = attn.expand_as(pred_f) * pred_f + (1 - attn.expand_as(pred_b)) * pred_b
        return pred, pred_f, m_mask_f, attn, pred_b, m_mask_b, 1 - attn


class UNetBidirection3(nn.Module):
    def __init__(self, im_height, im_width, im_channel, n_inputs, n_class, m_range, m_kernel):
        super(UNetBidirection3, self).__init__()
        num_hidden = 80
        self.bn0 = nn.BatchNorm2d(im_channel * n_inputs)
        self.conv0_1 = nn.Conv2d(im_channel * n_inputs, num_hidden, 3, 1, 1, bias=False)
        self.bn0_1 = nn.BatchNorm2d(num_hidden)
        # self.conv0_2 = nn.Conv2d(num_hidden, num_hidden, 3, 1, 1, bias=False)
        # self.bn0_2 = nn.BatchNorm2d(num_hidden)
        # self.conv0_3 = nn.Conv2d(num_hidden, num_hidden, 3, 1, 1, bias=False)
        # self.bn0_3 = nn.BatchNorm2d(num_hidden)
        self.conv1 = nn.Conv2d(num_hidden, num_hidden, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_hidden)
        self.conv2 = nn.Conv2d(num_hidden, num_hidden, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_hidden)
        self.conv3 = nn.Conv2d(num_hidden, num_hidden, 3, 1, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(num_hidden)
        self.conv4 = nn.Conv2d(num_hidden, num_hidden, 3, 1, 1, bias=False)
        self.bn4 = nn.BatchNorm2d(num_hidden)
        self.conv5 = nn.Conv2d(num_hidden, num_hidden, 3, 1, 1, bias=False)
        self.bn5 = nn.BatchNorm2d(num_hidden)
        self.conv6 = nn.Conv2d(num_hidden, num_hidden, 3, 1, 1, bias=False)
        self.bn6 = nn.BatchNorm2d(num_hidden)
        self.conv7 = nn.Conv2d(num_hidden, num_hidden, 3, 1, 1, bias=False)
        self.bn7 = nn.BatchNorm2d(num_hidden)
        self.conv8 = nn.Conv2d(num_hidden*2, num_hidden, 3, 1, 1, bias=False)
        self.bn8 = nn.BatchNorm2d(num_hidden)
        self.conv9 = nn.Conv2d(num_hidden*2, num_hidden, 3, 1, 1, bias=False)
        self.bn9 = nn.BatchNorm2d(num_hidden)
        self.conv10 = nn.Conv2d(num_hidden*2, num_hidden, 3, 1, 1, bias=False)
        self.bn10 = nn.BatchNorm2d(num_hidden)
        self.conv11 = nn.Conv2d(num_hidden*2, num_hidden, 3, 1, 1, bias=False)
        self.bn11 = nn.BatchNorm2d(num_hidden)
        self.conv12 = nn.Conv2d(num_hidden*2, num_hidden, 3, 1, 1, bias=False)
        self.bn12 = nn.BatchNorm2d(num_hidden)
        self.conv = nn.Conv2d(num_hidden * 2, num_hidden, 3, 1, 1)
        self.bn = nn.BatchNorm2d(num_hidden)

        self.conv2_1 = nn.Conv2d(num_hidden * 2, num_hidden, 3, 1, 1, bias=False)
        self.bn2_1 = nn.BatchNorm2d(num_hidden)
        self.conv2_2 = nn.Conv2d(num_hidden, num_hidden, 3, 1, 1, bias=False)
        self.bn2_2 = nn.BatchNorm2d(num_hidden)
        self.conv2_3 = nn.Conv2d(num_hidden, num_hidden, 3, 1, 1, bias=False)
        self.bn2_3 = nn.BatchNorm2d(num_hidden)
        self.conv2_4 = nn.Conv2d(num_hidden, num_hidden, 3, 1, 1, bias=False)
        self.bn2_4 = nn.BatchNorm2d(num_hidden)
        self.conv2_5 = nn.Conv2d(num_hidden, num_hidden, 3, 1, 1, bias=False)
        self.bn2_5 = nn.BatchNorm2d(num_hidden)
        self.conv2_6 = nn.Conv2d(num_hidden, num_hidden, 3, 1, 1, bias=False)
        self.bn2_6 = nn.BatchNorm2d(num_hidden)
        self.conv2_7 = nn.Conv2d(num_hidden, num_hidden, 3, 1, 1, bias=False)
        self.bn2_7 = nn.BatchNorm2d(num_hidden)
        self.conv2_8 = nn.Conv2d(num_hidden*2, num_hidden, 3, 1, 1, bias=False)
        self.bn2_8 = nn.BatchNorm2d(num_hidden)
        self.conv2_9 = nn.Conv2d(num_hidden*2, num_hidden, 3, 1, 1, bias=False)
        self.bn2_9 = nn.BatchNorm2d(num_hidden)
        self.conv2_10 = nn.Conv2d(num_hidden*2, num_hidden, 3, 1, 1, bias=False)
        self.bn2_10 = nn.BatchNorm2d(num_hidden)
        self.conv2_11 = nn.Conv2d(num_hidden*2, num_hidden, 3, 1, 1, bias=False)
        self.bn2_11 = nn.BatchNorm2d(num_hidden)
        self.conv2_12 = nn.Conv2d(num_hidden*2, num_hidden, 3, 1, 1, bias=False)
        self.bn2_12 = nn.BatchNorm2d(num_hidden)
        self.conv_2 = nn.Conv2d(num_hidden*2, n_class, 3, 1, 1)

        self.maxpool = nn.MaxPool2d(2, stride=2, return_indices=False, ceil_mode=False)
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)

        self.im_height = im_height
        self.im_width = im_width
        self.im_channel = im_channel
        self.n_class = n_class
        self.m_range = m_range
        self.m_kernel = m_kernel

    def forward(self, im_input_f, im_input_b, ones):
        x = self.bn0(im_input_f)
        x = F.relu(self.bn0_1(self.conv0_1(x)))
        # x = F.relu(self.bn0_2(self.conv0_2(x)))
        # x = F.relu(self.bn0_3(self.conv0_3(x)))

        x1 = F.relu(self.bn1(self.conv1(x)))
        x2 = self.maxpool(x1)
        x2 = F.relu(self.bn2(self.conv2(x2)))
        x3 = self.maxpool(x2)
        x3 = F.relu(self.bn3(self.conv3(x3)))
        x4 = self.maxpool(x3)
        x4 = F.relu(self.bn4(self.conv4(x4)))
        x5 = self.maxpool(x4)
        x5 = F.relu(self.bn5(self.conv5(x5)))
        x6 = self.maxpool(x5)
        x6 = F.relu(self.bn6(self.conv6(x6)))
        x7 = self.maxpool(x6)
        x7 = F.relu(self.bn7(self.conv7(x7)))
        x7 = self.upsample(x7)
        x8 = torch.cat((x7, x6), 1)
        x8 = F.relu(self.bn8(self.conv8(x8)))
        x8 = self.upsample(x8)
        x9 = torch.cat((x8, x5), 1)
        x9 = F.relu(self.bn9(self.conv9(x9)))
        x9 = self.upsample(x9)
        x10 = torch.cat((x9, x4), 1)
        x10 = F.relu(self.bn10(self.conv10(x10)))
        x10 = self.upsample(x10)
        x11 = torch.cat((x10, x3), 1)
        x11 = F.relu(self.bn11(self.conv11(x11)))
        x11 = self.upsample(x11)
        x12 = torch.cat((x11, x2), 1)
        x12 = F.relu(self.bn12(self.conv12(x12)))
        x12 = self.upsample(x12)
        y = torch.cat((x12, x1), 1)
        y = F.relu(self.bn(self.conv(y)))

        x = torch.cat((x, y), 1)
        x1 = F.relu(self.bn2_1(self.conv2_1(x)))
        x2 = self.maxpool(x1)
        x2 = F.relu(self.bn2_2(self.conv2_2(x2)))
        x3 = self.maxpool(x2)
        x3 = F.relu(self.bn2_3(self.conv2_3(x3)))
        x4 = self.maxpool(x3)
        x4 = F.relu(self.bn2_4(self.conv2_4(x4)))
        x5 = self.maxpool(x4)
        x5 = F.relu(self.bn2_5(self.conv2_5(x5)))
        x6 = self.maxpool(x5)
        x6 = F.relu(self.bn2_6(self.conv2_6(x6)))
        x7 = self.maxpool(x6)
        x7 = F.relu(self.bn2_7(self.conv2_7(x7)))
        x7 = self.upsample(x7)
        x8 = torch.cat((x7, x6), 1)
        x8 = F.relu(self.bn2_8(self.conv2_8(x8)))
        x8 = self.upsample(x8)
        x9 = torch.cat((x8, x5), 1)
        x9 = F.relu(self.bn2_9(self.conv2_9(x9)))
        x9 = self.upsample(x9)
        x10 = torch.cat((x9, x4), 1)
        x10 = F.relu(self.bn2_10(self.conv2_10(x10)))
        x10 = self.upsample(x10)
        x11 = torch.cat((x10, x3), 1)
        x11 = F.relu(self.bn2_11(self.conv2_11(x11)))
        x11 = self.upsample(x11)
        x12 = torch.cat((x11, x2), 1)
        x12 = F.relu(self.bn2_12(self.conv2_12(x12)))
        x12 = self.upsample(x12)
        y = torch.cat((x12, x1), 1)

        motion_f = self.conv_2(y)

        x = self.bn0(im_input_b)
        x = F.relu(self.bn0_1(self.conv0_1(x)))
        # x = F.relu(self.bn0_2(self.conv0_2(x)))
        # x = F.relu(self.bn0_3(self.conv0_3(x)))

        x1 = F.relu(self.bn1(self.conv1(x)))
        x2 = self.maxpool(x1)
        x2 = F.relu(self.bn2(self.conv2(x2)))
        x3 = self.maxpool(x2)
        x3 = F.relu(self.bn3(self.conv3(x3)))
        x4 = self.maxpool(x3)
        x4 = F.relu(self.bn4(self.conv4(x4)))
        x5 = self.maxpool(x4)
        x5 = F.relu(self.bn5(self.conv5(x5)))
        x6 = self.maxpool(x5)
        x6 = F.relu(self.bn6(self.conv6(x6)))
        x7 = self.maxpool(x6)
        x7 = F.relu(self.bn7(self.conv7(x7)))
        x7 = self.upsample(x7)
        x8 = torch.cat((x7, x6), 1)
        x8 = F.relu(self.bn8(self.conv8(x8)))
        x8 = self.upsample(x8)
        x9 = torch.cat((x8, x5), 1)
        x9 = F.relu(self.bn9(self.conv9(x9)))
        x9 = self.upsample(x9)
        x10 = torch.cat((x9, x4), 1)
        x10 = F.relu(self.bn10(self.conv10(x10)))
        x10 = self.upsample(x10)
        x11 = torch.cat((x10, x3), 1)
        x11 = F.relu(self.bn11(self.conv11(x11)))
        x11 = self.upsample(x11)
        x12 = torch.cat((x11, x2), 1)
        x12 = F.relu(self.bn12(self.conv12(x12)))
        x12 = self.upsample(x12)
        y = torch.cat((x12, x1), 1)
        y = F.relu(self.bn(self.conv(y)))

        x = torch.cat((x, y), 1)
        x1 = F.relu(self.bn2_1(self.conv2_1(x)))
        x2 = self.maxpool(x1)
        x2 = F.relu(self.bn2_2(self.conv2_2(x2)))
        x3 = self.maxpool(x2)
        x3 = F.relu(self.bn2_3(self.conv2_3(x3)))
        x4 = self.maxpool(x3)
        x4 = F.relu(self.bn2_4(self.conv2_4(x4)))
        x5 = self.maxpool(x4)
        x5 = F.relu(self.bn2_5(self.conv2_5(x5)))
        x6 = self.maxpool(x5)
        x6 = F.relu(self.bn2_6(self.conv2_6(x6)))
        x7 = self.maxpool(x6)
        x7 = F.relu(self.bn2_7(self.conv2_7(x7)))
        x7 = self.upsample(x7)
        x8 = torch.cat((x7, x6), 1)
        x8 = F.relu(self.bn2_8(self.conv2_8(x8)))
        x8 = self.upsample(x8)
        x9 = torch.cat((x8, x5), 1)
        x9 = F.relu(self.bn2_9(self.conv2_9(x9)))
        x9 = self.upsample(x9)
        x10 = torch.cat((x9, x4), 1)
        x10 = F.relu(self.bn2_10(self.conv2_10(x10)))
        x10 = self.upsample(x10)
        x11 = torch.cat((x10, x3), 1)
        x11 = F.relu(self.bn2_11(self.conv2_11(x11)))
        x11 = self.upsample(x11)
        x12 = torch.cat((x11, x2), 1)
        x12 = F.relu(self.bn2_12(self.conv2_12(x12)))
        x12 = self.upsample(x12)
        y = torch.cat((x12, x1), 1)

        motion_b = self.conv_2(y)

        m_mask_f = F.softmax(motion_f)
        m_mask_b = F.softmax(motion_b)

        seg_f = construct_seg(ones, m_mask_f, self.m_kernel, self.m_range)
        seg_b = construct_seg(ones, m_mask_b, self.m_kernel, self.m_range)

        pred_f = construct_image(im_input_f[:, -self.im_channel:, :, :], m_mask_f, self.m_kernel, self.m_range)
        pred_b = construct_image(im_input_b[:, -self.im_channel:, :, :], m_mask_b, self.m_kernel, self.m_range)

        attn = (seg_f + 1e-5) / (seg_f + seg_b + 2e-5)
        pred = attn.expand_as(pred_f) * pred_f + (1 - attn.expand_as(pred_b)) * pred_b
        return pred, m_mask_f, attn, m_mask_b, 1 - attn



class PreActBottleneck(nn.Module):
    def __init__(self, in_planes, planes):
        super(PreActBottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1_1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv1_2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=2, dilation=2, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x):
        out = F.relu(self.bn1(x))
        out1 = self.conv1_1(out)
        out2 = self.conv1_2(out)
        out = out1 + out2
        out = self.conv2(F.relu(self.bn2(out)))
        out += x
        return out


class DilateResNetBidirection(nn.Module):
    def __init__(self, im_height, im_width, im_channel, n_inputs, n_class, m_range, m_kernel, n_iter=12):
        super(DilateResNetBidirection, self).__init__()
        num_hidden = 64
        self.bn0 = nn.BatchNorm2d(im_channel * n_inputs)
        self.conv0 = nn.Conv2d(im_channel * n_inputs, num_hidden, 3, 1, 1, bias=False)
        self.layer = self._make_layer(PreActBottleneck, num_hidden, 10)
        self.bn = nn.BatchNorm2d(num_hidden)
        self.conv = nn.Conv2d(num_hidden, n_class, 3, 1, 1)
        self.im_height = im_height
        self.im_width = im_width
        self.im_channel = im_channel
        self.n_class = n_class
        self.m_range = m_range
        self.m_kernel = m_kernel
        self.n_iter = n_iter

    def _make_layer(self, block, planes, num_blocks):
        layers = []
        for i in range(num_blocks):
            layers.append(block(planes, planes))
        return nn.Sequential(*layers)

    def forward(self, im_input_f, im_input_b, ones):
        x = self.conv0(self.bn0(im_input_f))
        x = self.layer(x)
        motion_f = self.conv(F.relu(self.bn(x)))

        x = self.conv0(self.bn0(im_input_b))
        x = self.layer(x)
        motion_b = self.conv(F.relu(self.bn(x)))

        m_mask_f = F.softmax(motion_f)
        m_mask_b = F.softmax(motion_b)

        seg_f = construct_seg(ones, m_mask_f, self.m_kernel, self.m_range)
        seg_b = construct_seg(ones, m_mask_b, self.m_kernel, self.m_range)

        pred_f = construct_image(im_input_f[:, -self.im_channel:, :, :], m_mask_f, self.m_kernel, self.m_range)
        pred_b = construct_image(im_input_b[:, -self.im_channel:, :, :], m_mask_b, self.m_kernel, self.m_range)

        attn = (seg_f + 1e-5) / (seg_f + seg_b + 2e-5)
        pred = attn.expand_as(pred_f) * pred_f + (1 - attn.expand_as(pred_b)) * pred_b
        return pred, pred_f, m_mask_f, attn, pred_b, m_mask_b, 1 - attn


class UNetBidirection4(nn.Module):
    def __init__(self, im_height, im_width, im_channel, n_inputs, n_class, m_range, m_kernel):
        super(UNetBidirection4, self).__init__()
        num_hidden = 128
        self.conv0 = nn.Conv2d(im_channel * n_inputs, num_hidden, 3, 1, 1)
        self.bn0 = nn.BatchNorm2d(num_hidden)
        self.conv1 = nn.Conv2d(num_hidden, num_hidden, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(num_hidden)
        self.conv2 = nn.Conv2d(num_hidden, num_hidden, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(num_hidden)
        self.conv3 = nn.Conv2d(num_hidden, num_hidden, 3, 1, 1)
        self.bn3 = nn.BatchNorm2d(num_hidden)
        self.conv4 = nn.Conv2d(num_hidden, num_hidden, 3, 1, 1)
        self.bn4 = nn.BatchNorm2d(num_hidden)
        self.conv5 = nn.Conv2d(num_hidden, num_hidden, 3, 1, 1)
        self.bn5 = nn.BatchNorm2d(num_hidden)
        self.conv6 = nn.Conv2d(num_hidden, num_hidden, 3, 1, 1)
        self.bn6 = nn.BatchNorm2d(num_hidden)
        self.conv7 = nn.Conv2d(num_hidden, num_hidden, 3, 1, 1)
        self.bn7 = nn.BatchNorm2d(num_hidden)
        self.conv8 = nn.Conv2d(num_hidden*2, num_hidden, 3, 1, 1)
        self.bn8 = nn.BatchNorm2d(num_hidden)
        self.conv9 = nn.Conv2d(num_hidden*2, num_hidden, 3, 1, 1)
        self.bn9 = nn.BatchNorm2d(num_hidden)
        self.conv10 = nn.Conv2d(num_hidden*2, num_hidden, 3, 1, 1)
        self.bn10 = nn.BatchNorm2d(num_hidden)
        self.conv11 = nn.Conv2d(num_hidden*2, num_hidden, 3, 1, 1)
        self.bn11 = nn.BatchNorm2d(num_hidden)
        self.conv12 = nn.Conv2d(num_hidden*2, num_hidden, 3, 1, 1)
        self.bn12 = nn.BatchNorm2d(num_hidden)
        self.conv = nn.Conv2d(num_hidden * 2, num_hidden, 3, 1, 1)

        self.conv2_1 = nn.Conv2d(num_hidden * 2, num_hidden, 3, 1, 1)
        self.bn2_1 = nn.BatchNorm2d(num_hidden)
        self.conv2_2 = nn.Conv2d(num_hidden, num_hidden, 3, 1, 1)
        self.bn2_2 = nn.BatchNorm2d(num_hidden)
        self.conv2_3 = nn.Conv2d(num_hidden, num_hidden, 3, 1, 1)
        self.bn2_3 = nn.BatchNorm2d(num_hidden)
        self.conv2_4 = nn.Conv2d(num_hidden, num_hidden, 3, 1, 1)
        self.bn2_4 = nn.BatchNorm2d(num_hidden)
        self.conv2_5 = nn.Conv2d(num_hidden, num_hidden, 3, 1, 1)
        self.bn2_5 = nn.BatchNorm2d(num_hidden)
        self.conv2_6 = nn.Conv2d(num_hidden, num_hidden, 3, 1, 1)
        self.bn2_6 = nn.BatchNorm2d(num_hidden)
        self.conv2_7 = nn.Conv2d(num_hidden, num_hidden, 3, 1, 1)
        self.bn2_7 = nn.BatchNorm2d(num_hidden)
        self.conv2_8 = nn.Conv2d(num_hidden*2, num_hidden, 3, 1, 1)
        self.bn2_8 = nn.BatchNorm2d(num_hidden)
        self.conv2_9 = nn.Conv2d(num_hidden*2, num_hidden, 3, 1, 1)
        self.bn2_9 = nn.BatchNorm2d(num_hidden)
        self.conv2_10 = nn.Conv2d(num_hidden*2, num_hidden, 3, 1, 1)
        self.bn2_10 = nn.BatchNorm2d(num_hidden)
        self.conv2_11 = nn.Conv2d(num_hidden*2, num_hidden, 3, 1, 1)
        self.bn2_11 = nn.BatchNorm2d(num_hidden)
        self.conv2_12 = nn.Conv2d(num_hidden*2, num_hidden, 3, 1, 1)
        self.bn2_12 = nn.BatchNorm2d(num_hidden)
        self.conv_2 = nn.Conv2d(num_hidden*2, n_class, 3, 1, 1)

        self.maxpool = nn.MaxPool2d(2, stride=2, return_indices=False, ceil_mode=False)
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)

        self.im_height = im_height
        self.im_width = im_width
        self.im_channel = im_channel
        self.n_class = n_class
        self.m_range = m_range
        self.m_kernel = m_kernel

    def forward(self, im_input_f, im_input_b, ones):
        x = self.bn0(self.conv0(im_input_f))

        x1 = F.relu(self.bn1(self.conv1(x)))
        x2 = self.maxpool(x1)
        x2 = F.relu(self.bn2(self.conv2(x2)))
        x3 = self.maxpool(x2)
        x3 = F.relu(self.bn3(self.conv3(x3)))
        x4 = self.maxpool(x3)
        x4 = F.relu(self.bn4(self.conv4(x4)))
        x5 = self.maxpool(x4)
        x5 = F.relu(self.bn5(self.conv5(x5)))
        x6 = self.maxpool(x5)
        x6 = F.relu(self.bn6(self.conv6(x6)))
        x7 = self.maxpool(x6)
        x7 = F.relu(self.bn7(self.conv7(x7)))
        x7 = self.upsample(x7)
        x8 = torch.cat((x7, x6), 1)
        x8 = F.relu(self.bn8(self.conv8(x8)))
        x8 = self.upsample(x8)
        x9 = torch.cat((x8, x5), 1)
        x9 = F.relu(self.bn9(self.conv9(x9)))
        x9 = self.upsample(x9)
        x10 = torch.cat((x9, x4), 1)
        x10 = F.relu(self.bn10(self.conv10(x10)))
        x10 = self.upsample(x10)
        x11 = torch.cat((x10, x3), 1)
        x11 = F.relu(self.bn11(self.conv11(x11)))
        x11 = self.upsample(x11)
        x12 = torch.cat((x11, x2), 1)
        x12 = F.relu(self.bn12(self.conv12(x12)))
        x12 = self.upsample(x12)
        y = torch.cat((x12, x1), 1)
        y = self.conv(y)

        x = torch.cat((x, y), 1)
        x1 = F.relu(self.bn2_1(self.conv2_1(x)))
        x2 = self.maxpool(x1)
        x2 = F.relu(self.bn2_2(self.conv2_2(x2)))
        x3 = self.maxpool(x2)
        x3 = F.relu(self.bn2_3(self.conv2_3(x3)))
        x4 = self.maxpool(x3)
        x4 = F.relu(self.bn2_4(self.conv2_4(x4)))
        x5 = self.maxpool(x4)
        x5 = F.relu(self.bn2_5(self.conv2_5(x5)))
        x6 = self.maxpool(x5)
        x6 = F.relu(self.bn2_6(self.conv2_6(x6)))
        x7 = self.maxpool(x6)
        x7 = F.relu(self.bn2_7(self.conv2_7(x7)))
        x7 = self.upsample(x7)
        x8 = torch.cat((x7, x6), 1)
        x8 = F.relu(self.bn2_8(self.conv2_8(x8)))
        x8 = self.upsample(x8)
        x9 = torch.cat((x8, x5), 1)
        x9 = F.relu(self.bn2_9(self.conv2_9(x9)))
        x9 = self.upsample(x9)
        x10 = torch.cat((x9, x4), 1)
        x10 = F.relu(self.bn2_10(self.conv2_10(x10)))
        x10 = self.upsample(x10)
        x11 = torch.cat((x10, x3), 1)
        x11 = F.relu(self.bn2_11(self.conv2_11(x11)))
        x11 = self.upsample(x11)
        x12 = torch.cat((x11, x2), 1)
        x12 = F.relu(self.bn2_12(self.conv2_12(x12)))
        x12 = self.upsample(x12)
        y = torch.cat((x12, x1), 1)

        motion_f = self.conv_2(y)

        x = self.bn0(self.conv0(im_input_b))
        x1 = F.relu(self.bn1(self.conv1(x)))
        x2 = self.maxpool(x1)
        x2 = F.relu(self.bn2(self.conv2(x2)))
        x3 = self.maxpool(x2)
        x3 = F.relu(self.bn3(self.conv3(x3)))
        x4 = self.maxpool(x3)
        x4 = F.relu(self.bn4(self.conv4(x4)))
        x5 = self.maxpool(x4)
        x5 = F.relu(self.bn5(self.conv5(x5)))
        x6 = self.maxpool(x5)
        x6 = F.relu(self.bn6(self.conv6(x6)))
        x7 = self.maxpool(x6)
        x7 = F.relu(self.bn7(self.conv7(x7)))
        x7 = self.upsample(x7)
        x8 = torch.cat((x7, x6), 1)
        x8 = F.relu(self.bn8(self.conv8(x8)))
        x8 = self.upsample(x8)
        x9 = torch.cat((x8, x5), 1)
        x9 = F.relu(self.bn9(self.conv9(x9)))
        x9 = self.upsample(x9)
        x10 = torch.cat((x9, x4), 1)
        x10 = F.relu(self.bn10(self.conv10(x10)))
        x10 = self.upsample(x10)
        x11 = torch.cat((x10, x3), 1)
        x11 = F.relu(self.bn11(self.conv11(x11)))
        x11 = self.upsample(x11)
        x12 = torch.cat((x11, x2), 1)
        x12 = F.relu(self.bn12(self.conv12(x12)))
        x12 = self.upsample(x12)
        y = torch.cat((x12, x1), 1)
        y = self.conv(y)

        x = torch.cat((x, y), 1)
        x1 = F.relu(self.bn2_1(self.conv2_1(x)))
        x2 = self.maxpool(x1)
        x2 = F.relu(self.bn2_2(self.conv2_2(x2)))
        x3 = self.maxpool(x2)
        x3 = F.relu(self.bn2_3(self.conv2_3(x3)))
        x4 = self.maxpool(x3)
        x4 = F.relu(self.bn2_4(self.conv2_4(x4)))
        x5 = self.maxpool(x4)
        x5 = F.relu(self.bn2_5(self.conv2_5(x5)))
        x6 = self.maxpool(x5)
        x6 = F.relu(self.bn2_6(self.conv2_6(x6)))
        x7 = self.maxpool(x6)
        x7 = F.relu(self.bn2_7(self.conv2_7(x7)))
        x7 = self.upsample(x7)
        x8 = torch.cat((x7, x6), 1)
        x8 = F.relu(self.bn2_8(self.conv2_8(x8)))
        x8 = self.upsample(x8)
        x9 = torch.cat((x8, x5), 1)
        x9 = F.relu(self.bn2_9(self.conv2_9(x9)))
        x9 = self.upsample(x9)
        x10 = torch.cat((x9, x4), 1)
        x10 = F.relu(self.bn2_10(self.conv2_10(x10)))
        x10 = self.upsample(x10)
        x11 = torch.cat((x10, x3), 1)
        x11 = F.relu(self.bn2_11(self.conv2_11(x11)))
        x11 = self.upsample(x11)
        x12 = torch.cat((x11, x2), 1)
        x12 = F.relu(self.bn2_12(self.conv2_12(x12)))
        x12 = self.upsample(x12)
        y = torch.cat((x12, x1), 1)

        motion_b = self.conv_2(y)

        m_mask_f = F.softmax(motion_f)
        m_mask_b = F.softmax(motion_b)

        seg_f = self.construct_seg(ones, m_mask_f, self.m_kernel, self.m_range)
        seg_b = self.construct_seg(ones, m_mask_b, self.m_kernel, self.m_range)

        disappear_f = F.relu(seg_f - 1)
        appear_f = F.relu(1 - disappear_f)
        disappear_b = F.relu(seg_b - 1)
        appear_b = F.relu(1 - disappear_b)

        pred_f = self.construct_image(im_input_f[:, -self.im_channel:, :, :], m_mask_f, appear_f, self.m_kernel, self.m_range)
        pred_b = self.construct_image(im_input_b[:, -self.im_channel:, :, :], m_mask_b, appear_b, self.m_kernel, self.m_range)

        seg_f = 1 - F.relu(1 - seg_f)
        seg_b = 1 - F.relu(1 - seg_b)

        attn = (seg_f + 1e-5) / (seg_f + seg_b + 2e-5)
        pred = attn.expand_as(pred_f) * pred_f + (1 - attn.expand_as(pred_b)) * pred_b
        return pred, m_mask_f, 1 - appear_f, attn, m_mask_b, 1 - appear_b, 1 - attn

    def construct_seg(self, ones, m_mask, m_kernel, m_range):
        ones_expand = ones.expand_as(m_mask) * m_mask
        seg = Variable(torch.Tensor(ones.size()))
        if torch.cuda.is_available():
            seg = seg.cuda()
        for i in range(ones.size(0)):
            seg[i, :, :, :] = F.conv2d(ones_expand[i, :, :, :].unsqueeze(0), m_kernel, None, 1, m_range)
        return seg

    def construct_image(self, im, m_mask, appear, m_kernel, m_range):
        im = im * appear.expand_as(im)
        pred = Variable(torch.Tensor(im.size()))
        if torch.cuda.is_available():
            pred = pred.cuda()
        for i in range(im.size(1)):
            im_expand = im[:, i, :, :].unsqueeze(1).expand_as(m_mask) * m_mask
            for j in range(im.size(0)):
                pred[j, i, :, :] = F.conv2d(im_expand[j, :, :, :].unsqueeze(0), m_kernel, None, 1, m_range)
        return pred

