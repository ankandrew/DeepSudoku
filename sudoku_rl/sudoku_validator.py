import numpy as np

EXPECTED_VALUES = np.arange(1, 10)
"""Values that each row, column, and subgrid should contain for the Sudoku grid to be valid."""


def block_shaped(arr: np.ndarray, n_rows: int, n_cols: int) -> np.ndarray:
    """
    Modified from https://stackoverflow.com/a/16858283

    Return an array of shape (n, n_rows, n_cols) where
    n * n_rows * n_cols = arr.size

    If arr is a 2D array, the returned array should look like n sub-blocks with
    each sub-block preserving the "physical" layout of arr.
    """
    h, w = arr.shape
    assert h % n_rows == 0, f"{h} rows is not evenly divisible by {n_rows}"
    assert w % n_cols == 0, f"{w} cols is not evenly divisible by {n_cols}"
    return arr.reshape(h // n_rows, n_rows, -1, n_cols).swapaxes(1, 2).reshape(-1, n_rows * n_cols)


def val_sub_grids(grid: np.ndarray) -> bool:
    """
    Validate each 3x3 subgrid of the 9x9 grid

    :param grid: grid of type ndarray filled with all the values
    :return: True if all sub-grids contains the number [0, 9] w/o dupes
    """
    # Change so that each row contains 3x3 grid
    grid = block_shaped(grid, 3, 3)
    # Sort
    grid = np.sort(grid, axis=1)
    # Compare if values are the same as [1, 2, ..., 9]
    mask = grid == EXPECTED_VALUES
    # All must be true
    return np.all(mask)


def validate_axis(grid: np.ndarray, axis: int) -> bool:
    """
    Validate that all rows(axis=1) or columns(axis=0)
    have values of [0, 9] without duplicates

    :param grid: Array containing the grid fulled with values
    :param axis: Number 0 or 1, whether to check column/row - wise
    :return: True if all columns/rows contains numbers from [0, 9] w/o dupes
    """
    if axis == 1:
        grid = np.sort(grid, axis=1)
        return np.all(grid == EXPECTED_VALUES)
    elif axis == 0:
        grid = np.sort(grid, axis=0)
        return np.all(grid == np.expand_dims(EXPECTED_VALUES, axis=0).T)
    else:
        raise ValueError("Axis must be equal to 0 or 1")


def has_no_dupes(arr: np.ndarray) -> bool:
    arr = arr[arr != 0]  # Ignore zeros
    return len(arr) == len(set(arr))  # No duplicates


def is_unsolved_sudoku_valid(sudoku: np.ndarray) -> bool:
    # Check rows and columns
    for i in range(9):
        row = sudoku[i, :]
        col = sudoku[:, i]
        if not has_no_dupes(row) or not has_no_dupes(col):
            return False

    # Check sub-grids
    for i in range(0, 9, 3):
        for j in range(0, 9, 3):
            subgrid = sudoku[i : i + 3, j : j + 3].flatten()
            if not has_no_dupes(subgrid):
                return False

    return True


def is_solved_sudoku_valid(grid: np.ndarray) -> bool:
    return val_sub_grids(grid) and validate_axis(grid, axis=0) and validate_axis(grid, axis=1)
