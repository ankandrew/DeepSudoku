import random
from typing import Tuple

import numpy as np


def _mask_9x9_sudoku(grid: np.ndarray) -> np.ndarray:
    grid_copy = grid.copy()
    keep_numbers = 18
    mask_numbers = 81 - keep_numbers
    erase_vol = np.concatenate(
        (np.ones(keep_numbers, dtype=np.int8), np.zeros(mask_numbers, dtype=np.int8))
    )
    np.random.shuffle(erase_vol)
    return grid_copy * erase_vol.reshape(9, 9)


def generate_9x9_sudoku(shuffle: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a 9x9 Sudoku grid and return a 2-len tuple with the complete sudoku and other masked
    (with 0's).
    """
    # 1. Generate 9 non-repeating random values from [0, 9]
    row_1 = np.random.choice(range(1, 10), 9, replace=False).astype(np.int8)
    # 2. Shift of the first line by three slots.
    row_2 = np.roll(row_1, -3)
    # 3. Shift of the second line by three slots.
    row_3 = np.roll(row_2, -3)
    # 4. Shift of the third by one slot.
    row_4 = np.roll(row_3, -1)
    row_5 = np.roll(row_4, -3)
    row_6 = np.roll(row_5, -3)
    # Repeat
    row_7 = np.roll(row_6, -1)
    row_8 = np.roll(row_7, -3)
    row_9 = np.roll(row_8, -3)
    grid = np.vstack(
        (
            row_1,
            row_2,
            row_3,
            row_4,
            row_5,
            row_6,
            row_7,
            row_8,
            row_9,
        )
    )
    if shuffle:
        # Shuffle should be done via groups
        # E.g: https://stackoverflow.com/a/56581709
        # Groups N=3
        m, n = grid.shape[0] // 3, grid.shape[1]
        # Shuffle row-group-wise
        np.random.shuffle(grid.reshape(m, -1, n))
        # Shuffle column-group-wise
        np.random.shuffle(grid.T.reshape(m, -1, n))
        grid = grid.T
    return grid, _mask_9x9_sudoku(grid)
