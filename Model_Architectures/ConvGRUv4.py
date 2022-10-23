import torch
import torch.nn as nn
from Important_Modules.TimeDistributed import TimeDistributed
from Important_Modules.Conv3D import Conv3d
from Important_Modules.Conv2D import Conv2d


def model_over_permute(x, model):
    x = torch.permute(x, (0, 2, 1, 3, 4))
    x = model(x)
    x = torch.permute(x, (0, 2, 1, 3, 4))
    return x


class ConvGRUv4(nn.Module):
    def __init__(self):
        super(ConvGRUv4, self).__init__()
        self.conv3d = nn.Sequential(
            Conv3d(in_channels=1, output_channels=16, kernel_size=(3, 3, 3), padding='same', bias=False),
            nn.Dropout(p=0.2),
            nn.LeakyReLU(inplace=True),
            nn.GroupNorm(8, 16),
            nn.AdaptiveMaxPool3d((8, 55, 55)),
        )
        self.conv2d = TimeDistributed(Conv2dBlock(), batch_first=True)
        self.gru = nn.GRU(input_size=1936, hidden_size=100, batch_first=True, num_layers=2)
        self.classifier = nn.Sequential(
            nn.Linear(800, 128),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(128, 1),
        )
        self._initialize_weights()

    def forward(self, x):
        # input: N, C, D, H, W
        x = self.conv3d(x)
        x = torch.permute(x, (0, 2, 1, 3, 4))
        x = self.conv2d(x)
        x = x.view(x.size(0), x.size(1), -1)
        x, _ = self.gru(x)
        x = torch.reshape(x, (x.size(0), -1))
        x = torch.squeeze(self.classifier(x))
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


class depthwise_separable_conv(nn.Module):
    def __init__(self, nin, nout, padding):
        super(depthwise_separable_conv, self).__init__()
        self.depthwise = nn.Conv2d(nin, nin, kernel_size=3, padding=padding, groups=nin)
        self.pointwise = nn.Conv2d(nin, nout, kernel_size=1, padding=padding)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out


class Conv2dBlock(nn.Module):
    def __init__(self):
        super(Conv2dBlock, self).__init__()
        self.conv2d = nn.Sequential(
            Conv2d(in_channels=16, out_channels=128, kernel_size=(5, 5), bias=False),
            nn.Dropout(p=0.2),
            nn.LeakyReLU(inplace=True),
            nn.GroupNorm(8, 128),
            nn.AdaptiveMaxPool2d((26, 26)),
            depthwise_separable_conv(128, 256, padding=1),
            nn.Dropout(p=0.25),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(256),
            nn.AdaptiveMaxPool2d((13, 13)),
            depthwise_separable_conv(256, 64, 'same'),
            nn.Dropout(p=0.2),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(64),
            depthwise_separable_conv(64, 16, 0),
            nn.Dropout(p=0.25),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(16),
        )
        self._initialize_weights()

    def forward(self, x):
        return self.conv2d(x)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.GroupNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
