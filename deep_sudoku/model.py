from torch import nn
import torch.nn.functional as F


class Conv2dBlock(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, k=3, s=3, p=0, bn=False, activation=True):
        super(Conv2dBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=k, stride=s, padding=p, bias=not bn)
        self.bn = nn.BatchNorm2d(out_channels) if bn else None
        self.activation = activation

    def forward(self, x):
        x = self.conv(x)
        if self.bn:
            x = self.bn(x)
        if self.activation:
            x = F.relu(x)
        return x


class SudokuModel(nn.Module):
    def __init__(self):
        super(SudokuModel, self).__init__()
        self.block1 = Conv2dBlock(1, 120, k=3, s=(3, 3), bn=True, activation=True)
        self.block2 = Conv2dBlock(120, 120, k=3, s=1, bn=True, activation=True)
        self.block3 = Conv2dBlock(120, 729, k=1, s=1, bn=False, activation=False)

    def forward(self, x):
        x = self.block1(x)  # Output -> (batch, 120, 3, 3)
        x = self.block2(x)  # Output -> (batch, 120, 1, 1)
        x = self.block3(x)  # Output -> (batch, 729, 1, 1)
        return x
