import torch.nn as nn
import Model_Architectures.resnext as resnext
import torch


class ResNextGRU(nn.Module):
    def __init__(self):
        super(ResNextGRU, self).__init__()
        self.features = resnext.resnext50(sample_size=160, sample_duration=16)
        self.gru = nn.GRU(input_size=25, hidden_size=16, num_layers=2)
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(p=0.6),
            nn.Linear(128, 64),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(p=0.6),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.reshape(x, (x.size(0), 16, -1))
        x, _ = self.gru(x)
        x = torch.reshape(x, (x.size(0), -1))
        x = torch.squeeze(self.classifier(x))
        return x
