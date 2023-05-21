"""
Test utils module
"""
import numpy as np
import pytest

from sudoku_rl import utils


@pytest.mark.parametrize(
    "sudoku_str, expected_sudoku",
    [
        (
            """
            0 8 0 0 3 2 0 0 1
            7 0 3 0 8 0 0 0 2
            5 0 0 0 0 7 0 3 0
            0 5 0 0 0 1 9 7 0
            6 0 0 7 0 9 0 0 8
            0 4 7 2 0 0 0 5 0
            0 2 0 6 0 0 0 0 9
            8 0 0 0 9 0 3 0 5
            3 0 0 8 2 0 0 1 0
        """,
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
                ],
                dtype=np.int8,
            ),
        ),
    ],
)
def test_is_unsolved_sudoku_valid(sudoku_str: str, expected_sudoku: np.ndarray) -> None:
    np.testing.assert_array_equal(utils.parse_sudoku_from_str(sudoku_str), expected_sudoku)
