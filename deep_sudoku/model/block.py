from torch import nn
import torch.nn.functional as F
from typing import List


class Conv2dBlock(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, k=3, s=1, p=0, batch_norm=False, activation=True):
        super(Conv2dBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=k, stride=s, padding=p, bias=not batch_norm)
        self.batch_norm = nn.BatchNorm2d(out_channels) if batch_norm else None
        self.activation = activation

    def forward(self, x):
        x = self.conv(x)
        if self.batch_norm:
            x = self.batch_norm(x)
        if self.activation:
            x = F.relu(x)
        return x


class Conv1dBlock(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, k=3, s=1, p=0, batch_norm=False, activation=True):
        super(Conv1dBlock, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=k, stride=s, padding=p, bias=not batch_norm)
        self.batch_norm = nn.BatchNorm1d(out_channels) if batch_norm else None
        self.activation = activation

    def forward(self, x):
        x = self.conv(x)
        if self.batch_norm:
            x = self.batch_norm(x)
        if self.activation:
            x = F.relu(x)
        return x


class FullyConnected(nn.Module):
    def __init__(self, layers: List, batch_norm: bool = False, dropout_rate: float = None, activation: bool = True):
        super(FullyConnected, self).__init__()
        self.layers = layers
        self.batch_norm = batch_norm
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.fc = self.__make_fc()

    def forward(self, x):
        return self.fc(x)

    def __make_fc(self):
        layers = []
        penult_layer_i = len(self.layers) - 1
        for i in range(penult_layer_i):
            # First-Mid case (has activation)
            if i < penult_layer_i - 1:
                # If batch_norm is true we avoid the bias term in Linear layer (is redundant)
                layers.append(nn.Linear(self.layers[i], self.layers[i + 1], bias=not self.batch_norm))
                if self.batch_norm:
                    layers.append(nn.BatchNorm1d(self.layers[i + 1]))
                if self.activation:
                    layers.append(nn.ReLU())
                if self.dropout_rate:
                    layers.append(nn.Dropout(self.dropout_rate))
            # Last case (no activation)
            else:
                layers.append(nn.Linear(self.layers[i], self.layers[i + 1]))
        return nn.Sequential(*layers)
