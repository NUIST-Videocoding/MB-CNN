import math
import torch
import torch.nn as nn


def default_conv(in_channels, out_channels, kernel_size,stride, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,stride,
        padding=(kernel_size//2), bias=bias)


class MeanShift(nn.Conv2d):
    def __init__(self, rgb_range, rgb_mean, rgb_std, sign=-1):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1)
        self.weight.data.div_(std.view(3, 1, 1, 1))
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean)
        self.bias.data.div_(std)
        self.requires_grad = False


class BasicBlock(nn.Sequential):
    def __init__(
        self, in_channels, out_channels, kernel_size, stride=1, bias=False,
        bn=True, act=nn.ReLU(True)):

        m = [nn.Conv2d(
            in_channels, out_channels, kernel_size,
            padding=(kernel_size//2), stride=stride, bias=bias)
        ]
        if bn: m.append(nn.BatchNorm2d(out_channels))
        if act is not None: m.append(act)
        super(BasicBlock, self).__init__(*m)


class BasicBlock2(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, stride=2, bias=False, bn=True, act=nn.ReLU(True)):
        n = [nn.Conv2d(
            in_channels, out_channels, kernel_size,
            padding=(kernel_size//2), stride=stride, bias=bias)
        ]
        if bn: n.append(nn.BatchNorm2d(out_channels))
        if act is not None: n.append(act)
        super(BasicBlock, self).__init__(*n)


class ResBlock(nn.Module):
    def __init__(self, conv, n_feat, kernel_size,stride,bias=True, bn=True, act=nn.ReLU(True)):

        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feat, n_feat, kernel_size,stride, bias=bias))
            if i == 1 and bn: m.append(nn.BatchNorm2d(n_feat))
            if i == 0: m.append(act)
        self.body1 = nn.Sequential(*m)

    def forward(self, x):
        res = self.body1(x)
        res += x
        return res


class ResBlock2(nn.Module):
    def __init__(self, conv, n_feat, kernel_size,stride,bias=True, bn=True, act=nn.ReLU(True)):

        super(ResBlock2, self).__init__()
        m = []
        n = []
        for i in range(2):
            if i == 0: n.append(conv(n_feat//2, n_feat, kernel_size,stride*2, bias=bias))
            if i == 0: n.append(act)
            if i == 1: m.append(conv(n_feat, n_feat, kernel_size,stride, bias=bias))
            if i == 1 and bn: m.append(nn.BatchNorm2d(n_feat))
            #  if i == 1: m.append(act)
        self.conv=nn.Conv2d(n_feat//2, n_feat, 1, stride = 2, bias = False)
        self.body2 = nn.Sequential(*n,*m)

    def forward(self, x):
        res = self.body2(x)
        x=self.conv(x)
        res += x
        return res


class _Residual_Block(nn.Module):
    def __init__(self,bn = True):
        super(_Residual_Block, self).__init__()

        self.bn = bn
        self.conv1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.PReLU(num_parameters=1,init=0.2)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        if bn:
            self.bn1 = nn.BatchNorm2d(64)
            self.bn2 = nn.BatchNorm2d(64)

    def forward(self, x):
        output = self.conv1(x)
        if self.bn:
            output =self.bn1(output)
        output = self.relu(output)
        output = self.conv2(output)
        if self.bn:
            output = self.bn2(output)
        output = torch.add(output,x)
        return output


class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feat, bn=False, act=False, bias=True):

        m = []
        if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feat, 4 * n_feat, 3, bias))
                m.append(nn.PixelShuffle(2))
                if bn: m.append(nn.BatchNorm2d(n_feat))
                if act: m.append(act())
        elif scale == 3:
            m.append(conv(n_feat, 9 * n_feat, 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn: m.append(nn.BatchNorm2d(n_feat))
            if act: m.append(act())
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)

