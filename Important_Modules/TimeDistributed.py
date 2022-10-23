import torch.nn as nn
import torch


class TimeDistributed(nn.Module):

    def __init__(self, module, batch_first=False):
        super(TimeDistributed, self).__init__()
        self.module = module
        self.batch_first = batch_first

    def forward(self, x):
        ''' x size: (batch_size, time_steps, in_channels, height, width) '''
        batch_size, time_steps, C, H, W = x.size()
        # c_in = x.view(batch_size * time_steps, C, H, W)
        c_in = torch.reshape(x, (batch_size * time_steps, C, H, W))
        c_out = self.module(c_in)
        r_in = c_out.view(batch_size, time_steps, c_out.size(1), c_out.size(2), c_out.size(3))
        if self.batch_first is False:
            r_in = r_in.permute(1, 0, 2)
        return r_in
