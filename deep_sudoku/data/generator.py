import numpy as np
from .validator import Validator
from typing import Tuple


class Generator(Validator):
    """
    9 x 9 sudoku generator
    Numpy implementation based on Yaling Zheng answer
    https://gamedev.stackexchange.com/a/138228
    """

    def __init__(self):
        super(Generator, self).__init__()
        # self.grid = np.zeros((9, 9), dtype=np.int8)

    def __call__(self):
        y = self.generate(shuffle=True)
        x = self.remove_numbers(y)
        return x, y

    def generate_dataset(self, n: int) -> Tuple[np.ndarray, np.ndarray]:
        x_l = []
        y_l = []
        for _ in range(n):
            y = self.generate(shuffle=True)
            x = self.remove_numbers(y)
            x_l.append(x)
            y_l.append(y)
        return np.asarray(x_l), np.asarray(y_l)

    def generate(self, shuffle: bool = True) -> np.ndarray:
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
        # Validate
        # self.validate()
        grid = np.vstack((
            row_1, row_2, row_3,
            row_4, row_5, row_6,
            row_7, row_8, row_9,
        ))
        if shuffle:
            # Shuffle should be done via groups
            # E.g: https://stackoverflow.com/a/56581709
            # Groups N=3
            m, n = grid.shape[0] // 3, grid.shape[1]
            np.random.shuffle(grid.reshape(m, -1, n))
        # Validate
        if not self.validate(grid):
            grid = None
        return grid

    @staticmethod
    def remove_numbers(grid: np.ndarray) -> np.ndarray:
        erase_vol = np.random.choice([0, 1], size=(9, 3, 3), p=[0.6, 0.4])
        grid = grid.reshape((9, 3, 3)) * erase_vol
        grid = grid.reshape((9, 9))
        return grid
