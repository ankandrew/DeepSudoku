import os.path
import random
from typing import Union

import numpy as np
import torch
from torch.utils.data import DataLoader

from deep_sudoku.data.dataset import SudokuDataset
from deep_sudoku.data.generator import Generator
from deep_sudoku.data.validator import Validator
from deep_sudoku.metric import grid_accuracy
from deep_sudoku.model import SudokuMLP
from deep_sudoku.transform import ToTensor


def seed_all(seed: int = 1234) -> None:
    """
    Make code reproducible, taken from:
    https://discuss.pytorch.org/t/reproducibility-with-all-the-bells-and-whistles/81097

    :param seed: Seed to use in numpy, torch and python random module
    """
    print("[ Using Seed : ", seed, " ]")
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def print_accuracies(y_hat: torch.Tensor, y: torch.Tensor = None) -> None:
    """
    Verify and show if predicted grids `y_hat` are valid
    solutions and compare also to ground truth `y` if passed

    :param y_hat: Predictions with shape (batch, 9, 9)
    :param y: Optional ground truth with shape (batch, 9, 9)
    """
    # Valid solutions (same as ground truth OR different)
    val = Validator()
    total = y_hat.size(0)
    valid_count = 0
    for grid in y_hat:
        if val(grid.numpy()):
            valid_count += 1
    total_valid = valid_count
    print(f'Valid solutions {total_valid} out of {total} ({(100 * total_valid / total):.2f}%)')
    # "Correct" solution (same as ground truth)
    if y is not None:
        correct_grids = grid_accuracy(y_hat, y, valid=False)
        print(f'"Correct" solutions {correct_grids} out of {total} ({(100 * correct_grids / total):.2f}%)')


def one_hot_matrix(x: torch.Tensor, calc_max: bool = False) -> torch.Tensor:
    if calc_max:
        ncols = int(x.max().item() + 1)
    else:
        # Sudoku uses (0, 9) where 0 represents blank
        ncols = 10
    out = torch.zeros((x.numel(), ncols), dtype=torch.uint8)
    out[torch.arange(x.numel()), x.reshape(-1)] = 1
    out = out.view(x.shape + (ncols,))
    out = torch.movedim(out, -1, 0)  # Make it channels first
    return out


def preprocess(x: Union[np.ndarray, torch.Tensor]):
    # Convert to tensor if ndarray
    if isinstance(x, np.ndarray):
        # Convert to one-hot
        x = ToTensor.one_hot_matrix(x)
        x = torch.from_numpy(x)
    else:
        x = one_hot_matrix(x)
    x = x.type(torch.float32)
    # Add batch dimension
    x = torch.unsqueeze(x, dim=0)
    return x


def predict(model, x: Union[np.ndarray, torch.Tensor], do_preprocess=True):
    # Preprocess input
    if do_preprocess:
        x = preprocess(x).contiguous()
    # Avoid gradient calculations
    with torch.no_grad():
        pred = model(x)
        pred = pred.reshape(-1, 9, 9, 9)
        _, pred = pred.max(dim=1)
        # Scale from [0, num_classes-1] -> [1, num_classes]
        pred += 1
    return pred


def parse_str(sudoku: str) -> np.ndarray:
    """
    Parse a sudoku from a string to a np array.

    Example:

    game = '''
          0 8 0 0 3 2 0 0 1
          7 0 3 0 8 0 0 0 2
          5 0 0 0 0 7 0 3 0
          0 5 0 0 0 1 9 7 0
          6 0 0 7 0 9 0 0 8
          0 4 7 2 0 0 0 5 0
          0 2 0 6 0 0 0 0 9
          8 0 0 0 9 0 3 0 5
          3 0 0 8 2 0 0 1 0
      '''
    array_2d = parse_str(game)

    :param sudoku: String representing a Sudoku where rows
    are separated by a new line
    :return: Numpy matrix representing the given Sudoku
    """
    sudoku = sudoku.replace(' ', '').replace('\n', '')
    return np.asarray([int(i) for i in sudoku], dtype=np.int8).reshape((9, 9))
