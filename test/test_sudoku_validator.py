"""
Test sudoku_validator module
"""
import numpy as np
import pytest

from sudoku_rl import sudoku_validator


@pytest.mark.parametrize(
    "sudoku_grid",
    [
        np.array(
            [
                [0, 8, 0, 0, 3, 2, 0, 0, 1],
                [7, 0, 3, 0, 8, 0, 0, 0, 2],
                [5, 0, 0, 0, 0, 7, 0, 3, 0],
                [0, 5, 0, 0, 0, 1, 9, 7, 0],
                [6, 0, 0, 7, 0, 9, 0, 0, 8],
                [0, 4, 7, 2, 0, 0, 0, 5, 0],
                [0, 2, 0, 6, 0, 0, 0, 0, 9],
                [8, 0, 0, 0, 9, 0, 3, 0, 5],
                [3, 0, 0, 8, 2, 0, 0, 1, 0],
            ]
        ),
        np.array(
            [
                [1, 7, 9, 5, 3, 4, 8, 2, 6],
                [3, 4, 5, 6, 8, 2, 7, 9, 1],
                [8, 2, 6, 1, 7, 9, 4, 5, 3],
                [9, 1, 7, 4, 5, 3, 6, 8, 2],
                [5, 3, 4, 2, 6, 8, 1, 7, 9],
                [6, 8, 2, 9, 1, 7, 3, 4, 5],
                [7, 9, 1, 3, 4, 5, 2, 6, 8],
                [4, 5, 3, 8, 2, 6, 9, 1, 7],
                [2, 6, 8, 7, 9, 1, 5, 3, 4],
            ]
        ),
    ],
)
def test_is_sudoku_valid(sudoku_grid: np.ndarray) -> None:
    assert sudoku_validator.is_sudoku_valid(sudoku_grid)


@pytest.mark.parametrize(
    "sudoku_grid",
    [
        np.array(
            [
                [2, 8, 0, 0, 3, 2, 0, 0, 1],  # First element of this row is invalid
                [7, 0, 3, 0, 8, 0, 0, 0, 2],
                [5, 0, 0, 0, 0, 7, 0, 3, 0],
                [0, 5, 0, 0, 0, 1, 9, 7, 0],
                [6, 0, 0, 7, 0, 9, 0, 0, 8],
                [0, 4, 7, 2, 0, 0, 0, 5, 0],
                [0, 2, 0, 6, 0, 0, 0, 0, 9],
                [8, 0, 0, 0, 9, 0, 3, 0, 5],
                [3, 0, 0, 8, 2, 0, 0, 1, 0],
            ]
        ),
        np.array(
            [
                [0, 7, 9, 0, 3, 0, 8, 2, 0],
                [3, 0, 0, 6, 8, 2, 7, 9, 0],
                [8, 0, 6, 0, 7, 9, 4, 5, 3],
                [0, 0, 7, 0, 0, 0, 0, 0, 2],
                [0, 0, 4, 0, 0, 0, 1, 7, 9],
                [0, 8, 2, 0, 1, 0, 3, 0, 5],
                [0, 9, 1, 3, 0, 5, 2, 0, 0],
                [4, 5, 3, 0, 0, 0, 0, 0, 0],
                [2, 0, 8, 0, 9, 0, 0, 3, 5],  # Last element of this row is invalid
            ]
        ),
        np.array(
            [
                [0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 2, 3],
                [0, 0, 0, 0, 0, 0, 4, 5, 6],
                [0, 0, 0, 0, 0, 0, 7, 8, 8],  # Last sub-grid is invalid
            ]
        ),
    ],
)
def test_is_sudoku_invalid(sudoku_grid: np.ndarray) -> None:
    assert not sudoku_validator.is_sudoku_valid(sudoku_grid)
