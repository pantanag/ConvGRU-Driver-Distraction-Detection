from Model_Architectures import resnext
import torch.nn as nn
import torch


class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        self.features = resnext.resnext50(sample_size=160, sample_duration=16)
        self.gru = nn.GRU(input_size=25, hidden_size=16, num_layers=2)

    def forward(self, x):
        x = self.features(x)
        x = torch.reshape(x, (x.size(0), 16, -1))
        x, _ = self.gru(x)
        x = torch.reshape(x, (x.size(0), -1))
        return x
