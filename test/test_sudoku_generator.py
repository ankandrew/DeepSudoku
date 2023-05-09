"""
Test sudoku_generator module
"""
import numpy as np

from sudoku_rl import sudoku_generator, sudoku_validator


def test_generate_9x9_sudoku_valid() -> None:
    for _ in range(1_000):
        solved_sudoku, _ = sudoku_generator.generate_9x9_sudoku()
        assert sudoku_validator.is_sudoku_valid(solved_sudoku)


def test_generate_9x9_sudoku_mask_valid() -> None:
    solved_sudoku, unsolved_sudoku = sudoku_generator.generate_9x9_sudoku()
    np.testing.assert_array_equal(np.sqrt(solved_sudoku * unsolved_sudoku), unsolved_sudoku)
