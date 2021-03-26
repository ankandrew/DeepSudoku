from torch import nn
from typing import List
from deep_sudoku.model.block import Conv2dBlock, Conv1dBlock, FullyConnected
import torch


class SudokuConv(nn.Module):
    def __init__(self, batch_norm: bool = True):
        super(SudokuConv, self).__init__()
        self.block1 = Conv2dBlock(1, 120, k=3, s=(3, 3), batch_norm=batch_norm, activation=True)
        self.block2 = Conv2dBlock(120, 120, k=3, s=1, batch_norm=batch_norm, activation=True)
        self.block3 = Conv2dBlock(120, 729, k=1, s=1, batch_norm=batch_norm, activation=False)

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


# TODO: Add column/row features
class MultiBranchSudoku(nn.Module):
    def __init__(self, input_channels: int = 10):
        super(MultiBranchSudoku, self).__init__()
        # Local branch (applied at each 3x3 local grid)
        self.nb_conv1 = Conv2dBlock(input_channels, 128, k=3, s=(3, 3), p=0, batch_norm=False, activation=True)
        self.nb_conv2 = Conv2dBlock(128, 128, k=1, s=1, p=0, batch_norm=False, activation=True)
        self.nb_conv3 = Conv2dBlock(128, 128, k=3, s=1, p=0, batch_norm=False, activation=True)  # From (batch, 128, 3, 3) to (batch, 128, 1, 1)
        # Global branch (applied to the 9x9 global grid)
        self.gb_conv1 = Conv2dBlock(input_channels, 128, k=(9, 9), s=1, p=0, batch_norm=False, activation=True)
        self.gb_conv2 = Conv2dBlock(128, 128, k=1, s=1, p=0, batch_norm=False, activation=True)  # Conceptully is equal to a FC
        # # Row branch (applied to each row)
        # self.rb_conv1 = Conv1dBlock(input_channels, 128, k=9, s=1, p=0, batch_norm=False, activation=True)
        # self.rb_conv2 = Conv1dBlock(input_channels, 128, k=9, s=1, p=0, batch_norm=False, activation=True)
        # # Column branch (applied to each column)
        # Merge branch
        self.mb_conv1 = Conv2dBlock(128*2, 64, k=1, s=1, p=0, batch_norm=False, activation=True)
        self.mb_conv2 = Conv2dBlock(64, 128, k=1, s=1, p=0, batch_norm=False, activation=True)
        self.mb_conv3 = Conv2dBlock(128, 9*9*9, k=1, s=1, p=0, batch_norm=False, activation=False)

    def forward(self, x):
        # Local branch
        x1 = self.nb_conv1(x)
        x1 = self.nb_conv2(x1)
        x1 = self.nb_conv3(x1)  # out -> (batch, channels, 1, 1)
        # Global branch
        x2 = self.gb_conv1(x)
        x2 = self.gb_conv2(x2)  # out -> (batch, channels, 1, 1)
        # Merge branches
        x3 = torch.cat([x1, x2], dim=1)
        x4 = self.mb_conv1(x3)
        x4 = self.mb_conv2(x4)
        x4 = self.mb_conv3(x4)
        return x4

# TEST
# from deep_sudoku.data.dataset import SudokuDataset
# from deep_sudoku.transform import ToTensor
# from torch.utils.data import DataLoader
#
# model = HybridSudoku()
# # inp = torch.randint(0, 2, (1, 1, 9, 9))
# dataset = SudokuDataset(n=1, transform=ToTensor(one_hot=True))
# loader = DataLoader(dataset, batch_size=1, shuffle=True)
# x, y = next(iter(loader))
# model(x)
# print()
