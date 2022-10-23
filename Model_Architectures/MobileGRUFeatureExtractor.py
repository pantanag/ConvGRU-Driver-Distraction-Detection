import torch
import torch.nn as nn
from Model_Architectures.mobilenetv2 import MobileNetV2


class MobileGRUFeatureExtractor(nn.Module):
    def __init__(self):
        super(MobileGRUFeatureExtractor, self).__init__()
        self.mobile = MobileNetV2(sample_size=160)
        self.gru = nn.GRU(input_size=25, hidden_size=16, num_layers=2)

    def forward(self, x):
        x = self.mobile(x)
        x = torch.reshape(x, (x.size(0), 16, -1))
        x, _ = self.gru(x)
        x = torch.reshape(x, (x.size(0), -1))
        return x