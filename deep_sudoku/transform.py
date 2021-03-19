import torch
import numpy as np
from typing import Tuple


class ToTensor:
    """
    Convert to tensorts X / y and Preprocesses X
    """

    def __init__(self, scale: list = None, one_hot: bool = False) -> None:
        if scale is None:
            self.scale = [0, 1]
        else:
            if len(scale) != 2:
                raise ValueError('Scale len must be 2')
            self.scale = scale

        self.one_hot = one_hot

    def __call__(self, x: np.ndarray, y: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.one_hot:
            x = torch.tensor(self.one_hot_matrix(x), dtype=torch.float32)
        else:
            # Convert to Tensors
            x = torch.tensor(x, dtype=torch.float32)
            # Re-scale x
            x = self.__normalize(x)
            # Add channel dimension
            x = torch.unsqueeze(x, dim=0)
        y -= 1  # Make y class index of range [0, C−1]
        y = torch.tensor(y, dtype=torch.long)
        return x, y

    @staticmethod
    def one_hot_matrix(x: np.ndarray) -> np.ndarray:
        """
        One-hots each row and col elements of a matrix
        x is converted from (row, cols) -> (one_hot, row, cols)
        Modified from: https://stackoverflow.com/a/36960495
        :param x: Array containing the 2D matrix, shape expected (row, cols)
        :return: 3D array containing one-hot coded elements in channels first format
        """
        ncols = int(x.max() + 1)
        out = np.zeros((x.size, ncols), dtype=np.uint8)
        out[np.arange(x.size), x.ravel()] = 1
        out.shape = x.shape + (ncols,)
        out = np.moveaxis(out, -1, 0)  # Make it channels first
        return out

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """
        Normalize values between [a, b] / [self.scale[0], [self.scale[1]]

        :param x: Tensor to normalize
        :return: Normalized tensor between [self.scale[0], [self.scale[1]]
        """
        # min_num, max_num = x.max().item(), x.min().item()
        # prob = (x - min_num) / (max_num - min_num)
        return (self.scale[1] - self.scale[0]) * (x / 9) + self.scale[0]
