"""
Game module
"""
from dataclasses import dataclass

import numpy as np


@dataclass(order=False)
class SudokuGame:
    """..."""

    grid: np.ndarray
    difficulty: float = 0.4
