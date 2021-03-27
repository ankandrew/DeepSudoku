import torch
from deep_sudoku.data.validator import Validator
from deep_sudoku.metric import grid_accuracy
from deep_sudoku.transform import ToTensor
from typing import Union
import numpy as np
import random

from deep_sudoku.model import SudokuMLP


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


def verify_solution(y_hat: torch.Tensor, y: torch.Tensor = None) -> None:
    """
    Verify if predicted grids `y_hat` are valid solutions
    and compare also to ground truth `y` if passed

    :param y_hat: Predictions with shape (batch, 9, 9)
    :param y: Optional ground truth with shape (batch, 9, 9)
    """
    # Valid solutions (same as ground truth OR different)
    val = Validator()
    total = y_hat.size(0)
    valid_count = 0
    for grid in y_hat:
        # Convert grid from [0, C-1] to [1, C]
        if val(grid.numpy() + 1):
            valid_count += 1
    total_valid = valid_count
    print(f'Valid solutions {total_valid} out of {total} ({(100 * total_valid / total):.2f}%)')
    # "Correct" solution (same as ground truth)
    if y:
        correct_grids = grid_accuracy(y_hat, y)
        print(f'"Correct" solutions {correct_grids} out of {total} ({(100 * correct_grids / total):.2f}%)')


def one_hot_matrix(x: torch.Tensor, calc_max: bool = False) -> torch.Tensor:
    if calc_max:
        ncols = int(x.max().item() + 1)
    else:
        # Sudoku uses (0, 9) where 0 represents blank
        ncols = 10
    out = torch.zeros((x.numel(), ncols), dtype=torch.uint8)
    out[torch.arange(x.numel()), x.view(-1)] = 1
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


def predict(model, x: Union[np.ndarray, torch.Tensor]):
    # Preprocess input
    x = preprocess(x).contiguous()
    # Avoid gradient calculations
    with torch.no_grad():
        pred = model(x)
        pred = pred.reshape(-1, 9, 9, 9)
        _, pred = pred.max(dim=1)
        pred += 1
    return pred


def parse_str(sudoku: str) -> np.ndarray:
    """
    Parse a sudoku from a string to a np array

    :param sudoku: String representing a Sudoku where rows
    are separated by a new line
    :return: Numpy matrix representing the given Sudoku
    """
    sudoku = sudoku.replace(' ', '').replace('\n', '')
    return np.asarray([int(i) for i in sudoku], dtype=np.int8).reshape((9, 9))

# Test

# a1 = np.array(
#     [
#         [0, 0, 7, 5, 0, 0, 0, 0, 0],
#         [0, 0, 0, 0, 1, 0, 0, 4, 0],
#         [8, 0, 5, 7, 0, 0, 0, 0, 0],
#         [6, 0, 9, 0, 0, 0, 0, 0, 0],
#         [0, 4, 0, 0, 6, 0, 0, 1, 0],
#         [0, 0, 0, 0, 0, 0, 5, 0, 3],
#         [0, 0, 0, 0, 0, 5, 2, 0, 8],
#         [0, 6, 0, 0, 3, 0, 0, 0, 0],
#         [0, 0, 0, 0, 0, 8, 3, 0, 0]
#     ], np.int8)
#
# a2 = np.array(
#     [
#         [9, 0, 0, 4, 0, 2, 0, 0, 5],
#         [0, 0, 2, 0, 0, 0, 7, 0, 0],
#         [0, 0, 0, 6, 8, 0, 0, 0, 0],
#         [1, 0, 0, 0, 0, 0, 2, 0, 4],
#         [0, 0, 9, 0, 0, 0, 8, 0, 0],
#         [5, 0, 3, 0, 0, 0, 0, 0, 6],
#         [0, 0, 0, 0, 6, 7, 0, 0, 0],
#         [0, 0, 4, 0, 0, 0, 9, 0, 0],
#         [6, 0, 0, 1, 0, 3, 0, 0, 7]
#     ], np.int8)
#
# a3 = np.array(
#     [
#         [0, 0, 0, 0, 5, 0, 0, 0, 0],
#         [0, 8, 7, 0, 3, 0, 0, 6, 0],
#         [0, 0, 0, 8, 0, 4, 0, 2, 0],
#         [0, 0, 5, 0, 0, 0, 4, 0, 0],
#         [2, 9, 0, 0, 0, 0, 0, 1, 7],
#         [0, 0, 6, 0, 0, 0, 9, 0, 0],
#         [0, 7, 0, 1, 0, 8, 0, 0, 0],
#         [0, 1, 0, 0, 6, 0, 5, 4, 0],
#         [0, 0, 0, 0, 2, 0, 0, 0, 0]
#     ], np.int8)
#
# a4 = np.array(
#     [
#         [0, 4, 0, 3, 0, 2, 0, 7, 8],
#         [0, 0, 0, 0, 0, 0, 0, 0, 0],
#         [9, 1, 0, 0, 0, 0, 0, 3, 5],
#         [3, 0, 0, 9, 0, 6, 0, 0, 1],
#         [0, 0, 0, 0, 8, 0, 0, 0, 0],
#         [2, 0, 0, 1, 0, 5, 0, 0, 9],
#         [1, 8, 0, 0, 0, 0, 0, 4, 6],
#         [0, 0, 0, 0, 0, 0, 0, 0, 0],
#         [0, 3, 0, 7, 0, 9, 0, 1, 0]
#     ], np.int8)
#
# # Load model
# model = SudokuMLP([10 * 9 * 9, 120, 120, 9 * 9 * 9], batch_norm=False, dropout_rate=0.5)
# model.load_state_dict(torch.load('C:/Users/Anka/Downloads/sudoku_model.pth'))
# # Set model to eval (turns off dropout e.g.)
# model.eval()
# res = predict(model, a4)
# print(res)
# # Check if is valid solution
# val = Validator()
# print(f' Is a valid ? {val(res[0].numpy())}')
# print()

# game = '''
#           0 8 0 0 3 2 0 0 1
#           7 0 3 0 8 0 0 0 2
#           5 0 0 0 0 7 0 3 0
#           0 5 0 0 0 1 9 7 0
#           6 0 0 7 0 9 0 0 8
#           0 4 7 2 0 0 0 5 0
#           0 2 0 6 0 0 0 0 9
#           8 0 0 0 9 0 3 0 5
#           3 0 0 8 2 0 0 1 0
#       '''
#
# print(parse_str(game))
