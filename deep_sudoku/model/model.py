from torch import nn
from typing import List
from deep_sudoku.model.block import Conv2dBlock, Conv1dBlock, FullyConnected
import torch

"""
TODO:
* Try Layer Norm in FC (Like transformers)
* Test ReLU variants in FC (i.e. leaky relu)
"""


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


class MultiBranchSudoku(nn.Module):
    def __init__(self, input_channels: int = 10):
        super(MultiBranchSudoku, self).__init__()
        # Local branch (applied at each 3x3 local grid)
        self.nb_conv1 = Conv2dBlock(input_channels, 128, k=3, s=(3, 3), p=0, batch_norm=False, activation=True)
        self.nb_conv2 = Conv2dBlock(128, 128, k=1, s=1, p=0, batch_norm=False, activation=True)
        self.nb_conv3 = Conv2dBlock(128, 128, k=3, s=1, p=0, batch_norm=False,
                                    activation=True)  # From (batch, 128, 3, 3) to (batch, 128, 1, 1)
        # Global branch (applied to the 9x9 global grid)
        self.gb_conv1 = Conv2dBlock(input_channels, 128, k=(9, 9), s=1, p=0, batch_norm=False, activation=True)
        self.gb_conv2 = Conv2dBlock(128, 128, k=1, s=1, p=0, batch_norm=False,
                                    activation=True)  # Conceptully is equal to a FC
        # # Row branch (applied to each row)
        self.rb_conv1 = Conv1dBlock(input_channels, 128, k=9, s=9, p=0, batch_norm=False, activation=True)
        # # Column branch (applied to each column)
        self.cb_conv1 = Conv1dBlock(input_channels, 128, k=9, s=9, p=0, batch_norm=False, activation=True)
        # Merge branch
        self.fc = FullyConnected([128 * 2 + 128 * 9 * 2, 512, 9 * 9 * 9], dropout_rate=0.5)

    def forward(self, x):
        # Local branch
        x1 = self.nb_conv1(x)
        x1 = self.nb_conv2(x1)
        x1 = self.nb_conv3(x1)  # out -> (batch, channels, 1, 1)
        x1 = x1.flatten(1, -1)
        # Global branch
        x2 = self.gb_conv1(x)
        x2 = self.gb_conv2(x2)  # out -> (batch, channels, 1, 1)
        x2 = x2.flatten(1, -1)
        # Row branch
        x3 = x.reshape(-1, x.size(1), x.size(2) ** 2)
        x3 = self.rb_conv1(x3)  # out shape -> (batch, 128, 9)
        x3 = x3.flatten(1, -1)
        # Col branch
        x4 = x.movedim(-2, -1)
        x4 = x4.reshape(-1, x.size(1), x.size(2) ** 2)
        x4 = self.cb_conv1(x4)  # out shape -> (batch, 128, 9)
        x4 = x4.flatten(1, -1)
        # Merge branches
        x5 = torch.cat([x1, x2, x3, x4], dim=1)
        x6 = self.fc(x5)
        return x6


class RNN(nn.Module):
    """
    Wraps around torch.nn recurrent nets variants. Optional applies a
    Fully Connected/Dense layer to each hidden state at timestep `t`.

    Examples
    --------
    >>> rnn = RNN(10, 128, 1, 'lstm', [256, 9], fc_bn=True, fc_dropout=0.25, bidirectional=True)
    >>> random_input = torch.randn((64, 81, 10), dtype=torch.float32)
    >>> out = rnn(random_input)
    >>> out.shape
    torch.Size([64, 81, 9])

    Number of classes is assumed from fc_layers last element (fc_layers[-1]).

    Note: Since in this example is bidirectional hidden_state is multiplied by 2,
    hence fc_layers[-1] must be hidden_state * 2
    """

    def __init__(self, input_size: int,
                 hidden_size: int,
                 n_layers: int,
                 rnn_type: str = 'lstm',
                 fc_layers: List[int] = None,
                 fc_bn: bool = False,
                 fc_dropout: float = None,
                 bidirectional: bool = False):
        super(RNN, self).__init__()
        if rnn_type == 'rnn':
            self.rnet = nn.RNN(input_size, hidden_size, n_layers, batch_first=True, bidirectional=bidirectional)
        elif rnn_type == 'lstm':
            self.rnet = nn.LSTM(input_size, hidden_size, n_layers, batch_first=True, bidirectional=bidirectional)
        elif rnn_type == 'gru':
            self.rnet = nn.GRU(input_size, hidden_size, n_layers, batch_first=True, bidirectional=bidirectional)
        else:
            raise ValueError(f'Aviable options are: gru/lstm/rnn but got {rnn_type}')
        # Cleaner than one-liner
        if fc_layers is not None:
            # Calculate hidden state based on bidirectional
            final_hidden_size = hidden_size * 2 if bidirectional else hidden_size
            # Check (final_hidden_size output features) == (fc_layers input features)
            if final_hidden_size != fc_layers[0]:
                raise ValueError('hidden state dimension must be equal to fc layers first element')
            self.fc = FullyConnected(fc_layers, batch_norm=fc_bn, dropout_rate=fc_dropout, activation=True)
        else:
            self.fc = None

    def forward(self, x):
        if isinstance(self.rnet, nn.LSTM):
            out, (h_n, _) = self.rnet(x)  # outs -> (batch, seq_len, hidden_size)
        else:
            out, h_n = self.rnet(x)  # outs -> (batch, seq_len, hidden_size)
        if self.fc is not None:
            out = self.fc(out)  # outs -> (batch, seq_len, num_classes/fc_layers[-1])
        return out
