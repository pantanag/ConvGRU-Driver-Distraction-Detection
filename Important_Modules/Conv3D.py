import torch.nn as nn
import torch.nn.functional as F


class Conv3d(nn.Conv3d):
    def __init__(self, in_channels, output_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(Conv3d, self).__init__(in_channels, output_channels, kernel_size, stride, padding, dilation, groups, bias)

    def forward(self, x):
        w = self.weight
        w_mean = w.mean(dim=1, keepdim=True).mean(dim=2, keepdim=True).mean(dim=3, keepdim=True).mean(dim=4, keepdim=True)
        w = w - w_mean
        std = w.view(w.size(0), -1).std(dim=1).view(-1, 1, 1, 1, 1) + 1e-5
        w = w / std.expand_as(w)
        w = w.cuda()
        return F.conv3d(x, w, self.bias, self.stride, self.padding, self.dilation, self.groups)
