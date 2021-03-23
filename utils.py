import torch
from deep_sudoku.data.validator import Validator
from deep_sudoku.metric import grid_accuracy


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
