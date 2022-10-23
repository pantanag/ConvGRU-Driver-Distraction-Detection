import torch
import torch.nn as nn



class pospowbias(nn.Module):
    def __init__(self):
        super(pospowbias, self).__init__()
        self.Lambda = nn.Parameter(torch.zeros(1))
        self.Alpha = nn.Parameter(torch.zeros(1))

    def forward(self, x):

        return PositivePowBias(x, self.Alpha, self.Lambda, options=best_options)

class DPP(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.pospowbias=pospowbias()
    def forward(self, I):
        It   = F.upsample(F.avg_pool2d(I, 2), scale_factor=2, mode='nearest')
        x   = ((I-It)**2)+1e-3
        xn = F.upsample(F.avg_pool2d(x, 2), scale_factor=2, mode='nearest')
        w  = pospowbias(x/xn)
        kp = F.avg_pool2d(w, 2)
        Iw = F.avg_pool2d(I*w, 2)
        return Iw/kp