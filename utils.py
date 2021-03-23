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
        if not val(grid.numpy()):
            valid_count += 1
    total_valid = valid_count
    print(f'Valid solutions {total_valid} out of {total} ({(100 * total_valid / total):.2f}%)')
    # "Correct" solution (same as ground truth)
    if y:
        correct_grids = grid_accuracy(y_hat, y)
        print(f'"Correct" solutions {correct_grids} out of {total} ({(100 * correct_grids / total):.2f}%)')


# # Test
#
# t = torch.tensor([[[0, 1, 7, 5, 2, 8, 4, 3, 6],
#                    [5, 2, 8, 4, 3, 6, 0, 1, 7],
#                    [4, 3, 6, 0, 1, 7, 5, 2, 8],
#                    [7, 6, 2, 8, 4, 3, 3, 0, 1],
#                    [8, 4, 3, 3, 0, 1, 7, 6, 2],
#                    [3, 0, 1, 7, 6, 2, 8, 4, 3],
#                    [3, 6, 4, 3, 7, 5, 2, 8, 4],
#                    [3, 7, 5, 2, 8, 4, 3, 6, 4],
#                    [2, 8, 4, 3, 6, 4, 3, 7, 5]],
#                   [[6, 7, 1, 3, 5, 4, 1, 0, 2],
#                    [3, 5, 4, 1, 0, 2, 6, 7, 1],
#                    [1, 0, 2, 6, 7, 1, 3, 5, 4],
#                    [4, 3, 0, 2, 6, 7, 1, 3, 5],
#                    [2, 6, 7, 1, 3, 5, 4, 3, 0],
#                    [1, 3, 5, 4, 3, 0, 2, 6, 7],
#                    [0, 2, 6, 7, 1, 3, 5, 4, 8],
#                    [7, 1, 3, 5, 4, 8, 0, 2, 6],
#                    [5, 4, 8, 0, 2, 6, 7, 1, 3]],
#                   [[6, 3, 8, 4, 1, 0, 7, 5, 2],
#                    [4, 1, 0, 7, 5, 2, 6, 3, 8],
#                    [7, 5, 2, 6, 3, 8, 4, 1, 0],
#                    [8, 4, 1, 2, 6, 5, 1, 6, 7],
#                    [2, 6, 5, 2, 6, 7, 8, 4, 1],
#                    [2, 6, 7, 8, 4, 1, 2, 6, 5],
#                    [5, 2, 6, 3, 8, 4, 1, 0, 7],
#                    [3, 8, 4, 1, 0, 7, 5, 2, 6],
#                    [1, 0, 7, 5, 2, 6, 3, 8, 4]]])
#
# verify_solution(t)
