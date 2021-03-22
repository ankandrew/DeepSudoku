from torch import nn
from typing import List
from deep_sudoku.model.block import Conv2dBlock, FullyConnected


class SudokuConv(nn.Module):
    def __init__(self):
        super(SudokuConv, self).__init__()
        self.block1 = Conv2dBlock(1, 120, k=3, s=(3, 3), batch_norm=True, activation=True)
        self.block2 = Conv2dBlock(120, 120, k=3, s=1, batch_norm=True, activation=True)
        self.block3 = Conv2dBlock(120, 729, k=1, s=1, batch_norm=False, activation=False)

    def forward(self, x):
        x = self.block1(x)  # Output -> (batch, 120, 3, 3)
        x = self.block2(x)  # Output -> (batch, 120, 1, 1)
        x = self.block3(x)  # Output -> (batch, 729, 1, 1)
        return x


class SudokuMLP(nn.Module):
    def __init__(self, layers: List, batch_norm: bool = False, dropout_rate: float = None):
        super(SudokuMLP, self).__init__()
        self.input_dim = layers[0]
        self.fc = FullyConnected(layers, batch_norm=batch_norm, dropout_rate=dropout_rate)

    def forward(self, x):
        x = x.view(-1, self.input_dim)  # (batch, c, h, w) -> (batch, c*h*w)
        x = self.fc(x)
        return x
