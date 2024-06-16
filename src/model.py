
from torch import nn
import torch.nn.functional as F
import torchvision

class Residual(nn.Module):
    def __init__(self, input_channels, min_channels, num_channels,
                 use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, min_channels,
                               kernel_size=1)
        self.conv2 = nn.Conv2d(min_channels, min_channels,
                               kernel_size=3, padding=1, stride=strides)
        self.conv3 = nn.Conv2d(min_channels, num_channels,
                               kernel_size=1)
        if use_1x1conv:
            self.conv4 = nn.Conv2d(input_channels, num_channels,
                                   kernel_size=1, stride=strides)
        else:
            self.conv4 = None
        self.bn1 = nn.BatchNorm2d(min_channels)
        self.bn2 = nn.BatchNorm2d(min_channels)
        self.bn3 = nn.BatchNorm2d(num_channels)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        Y = self.bn3(self.conv3(Y))
        if self.conv4:
            X = self.conv4(X)
        Y += X
        return F.relu(Y)


def resnet_block(input_channels, min_channels, num_channels, num_residuals, stride,
                 first_block=False):
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(Residual(input_channels, min_channels, num_channels,
                                use_1x1conv=True, strides=stride))
        elif first_block and i == 0:
            blk.append(Residual(input_channels, min_channels, num_channels, use_1x1conv=True))
        else:
            blk.append(Residual(num_channels, min_channels, num_channels))
    return blk


def ResNet():
    b1 = nn.Sequential(nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
                    nn.BatchNorm2d(64), nn.ReLU(),
                    nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
    b2 = nn.Sequential(*resnet_block(64, 64, 256, 3, 2, first_block=True))
    b3 = nn.Sequential(*resnet_block(256, 128, 512, 4, 2))
    b4 = nn.Sequential(*resnet_block(512, 256, 1024, 6, 2))
    b5 = nn.Sequential(*resnet_block(1024, 512, 2048, 2, 2))
    net = nn.Sequential(b1, b2, b3, b4, b5,
                        nn.AdaptiveAvgPool2d((1, 1)),
                        nn.Flatten(), nn.Linear(2048, 1588))
    return net
