import torch
import torch.nn as nn
from Model_Architectures.shufflenet import ShuffleNet


class shuffleGRU(nn.Module):
    def __init__(self):
        super(shuffleGRU, self).__init__()
        self.shufflenet = ShuffleNet(width_mult=2., groups=4)
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
        x = self.shufflenet(x)
        x = torch.reshape(x, (x.size(0), 16, -1))
        x, _ = self.gru(x)
        x = torch.reshape(x, (x.size(0), -1))
        x = torch.squeeze(self.classifier(x))
        # x = self.classifier(x)
        return x
