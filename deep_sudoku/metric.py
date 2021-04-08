import torch
from .data.validator import Validator


def accuracy(y_hat: torch.Tensor, y: torch.Tensor) -> float:
    """
    Calculates how many predictions were correct in each slot of each grid

    :param y_hat: Predicted values by de model
    :param y: Ground truth values
    :return: Amount of corrected predicted slots of each grid
    """
    return torch.eq(y_hat, y).sum().item()


def grid_accuracy(y_hat: torch.Tensor, y: torch.Tensor, valid: bool = True) -> float:
    """
    This metric tells how many grids were predicted correctly
    If a grid wass off by one value, it counts as a misprediction

    :param y_hat: Predicted values by de model of shape (batch, 9, 9)
    :param y: Ground truth values of shape (batch, 9, 9)
    :param valid: Consider as correct if it is different
    from ground truth but still a valid solution
    :return: Amount of perfectly predicted grids
    """
    mask = torch.eq(y_hat, y)
    b, h, w = mask.shape
    mask_2d = mask.view(b, h * w)
    final_mask = torch.all(mask_2d, dim=1)
    if valid:
        final_mask = final_mask.cpu()
        vals = []
        val = Validator()
        for grid in y_hat:
            vals.append(val(grid.cpu().numpy()))
        mask_valid = torch.tensor(vals, dtype=torch.bool)
        final_mask |= mask_valid
    return final_mask.sum().item()
