import logging
import random

import numpy as np

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)


def seed_all(seed: int = 1984) -> None:
    """
    Make code reproducible, taken from:
    https://discuss.pytorch.org/t/reproducibility-with-all-the-bells-and-whistles/81097

    :param seed: Seed to use in numpy, torch and python random module
    """
    LOGGER.info("[ Using Seed : %s ]", seed)
    np.random.seed(seed)
    random.seed(seed)


def parse_sudoku_from_str(sudoku_str: str) -> np.ndarray:
    """
    Parse a sudoku from a string to a np array.

    Example:

    game = '''
          0 8 0 0 3 2 0 0 1
          7 0 3 0 8 0 0 0 2
          5 0 0 0 0 7 0 3 0
          0 5 0 0 0 1 9 7 0
          6 0 0 7 0 9 0 0 8
          0 4 7 2 0 0 0 5 0
          0 2 0 6 0 0 0 0 9
          8 0 0 0 9 0 3 0 5
          3 0 0 8 2 0 0 1 0
      '''
    array_2d = parse_sudoku_from_str(game)

    :param sudoku_str: String representing a Sudoku where rows are separated by a new line.
    :return: Numpy matrix representing the given Sudoku.
    """
    sudoku_str = sudoku_str.replace(" ", "").replace("\n", "")
    return np.asarray([int(i) for i in sudoku_str], dtype=np.int8).reshape((9, 9))


def sudoku_9x9_to_str(grid: np.ndarray) -> str:
    """
    Return a String representation of the given 9x9 Sudoku grid.

    :param grid: Numpy array with the Sudoku grid.
    :return: String representation of the given Sudoku.
    """
    sudoku_str = ""
    for i in range(9):
        if i % 3 == 0 and i != 0:
            sudoku_str += "- - - - - - - - - - - \n"
        for j in range(9):
            if j % 3 == 0 and j != 0:
                sudoku_str += "| "
            if j == 8:
                sudoku_str += f"{grid[i][j]}\n"
            else:
                sudoku_str += f"{grid[i][j]} "
    return sudoku_str
