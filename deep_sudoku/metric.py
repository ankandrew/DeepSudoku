import torch
# TODO: In grid_accuracy add Validator
#       (In the case it is correct but different to ground truth):
# from .validator import Validator


def accuracy(y_hat: torch.Tensor, y: torch.Tensor) -> float:
    """
    Calculates how many predictions were correct in each slot of each grid

    :param y_hat: Predicted values by de model
    :param y: Ground truth values
    :return: Amount of corrected predicted slots of each grid
    """
    return torch.eq(y_hat, y).sum().item()


def grid_accuracy(y_hat: torch.Tensor, y: torch.Tensor) -> float:
    """
    This metric tells how many grids were predicted correctly
    If it wass off by one values, it counts as a misprediction

    :param y_hat: Predicted values by de model
    :param y: Ground truth values
    :return: Amount of perfectly predicted grids
    """
    mask = torch.eq(y_hat, y)
    b, h, w = mask.shape
    mask_2d = mask.view(b, h * w)
    return torch.all(mask_2d, dim=1).sum().item()


# Debug

# test_y_hat = torch.tensor([
#     [
#         [1, 2, 3],
#         [4, 5, 6],
#         [7, 8, 9]
#     ],
#     [
#         [10, 11, 12],
#         [13, 14, 15],
#         [16, 17, 18]
#     ]
# ], dtype=torch.float32)
#
# test_y = torch.tensor([
#     [
#         [1, 2, 3],
#         [4, 5, 6],
#         [7, 8, 9]
#     ],
#     [
#         [10, 11, 100000],  # Diff. element
#         [13, 14, 15],
#         [16, 17, 18]
#     ]
# ], dtype=torch.float32)
#
# out = grid_accuracy(test_y_hat, test_y)
