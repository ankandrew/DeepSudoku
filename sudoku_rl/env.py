"""
Check: https://www.gymlibrary.dev/content/environment_creation/
"""

from enum import IntEnum
from typing import Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from sudoku_rl import sudoku_generator, sudoku_validator


class SudokuReward(IntEnum):
    WIN = 1.0
    VALID_ACTION = 0.0
    INVALID_ACTION = -0.1


class SudokuEnv(gym.Env):
    """Custom Sudoku Environment that follows gym interface."""

    metadata = {"render.modes": ["human"]}

    def __init__(self):
        super().__init__()
        self.action_space = spaces.Discrete(9 * 9 * 9)
        self.observation_space = spaces.Box(low=0, high=9, shape=(9 * 9,), dtype=np.int8)
        self.solved_grid, self.play_grid = sudoku_generator.generate_9x9_sudoku()

    def _get_info(self):
        return {"filled_cells": np.count_nonzero(self.play_grid)}

    def _get_obs(self):
        return self.play_grid.flatten()

    def _play_action(self, action: int) -> Tuple[SudokuReward, bool]:
        if 0 > action > 728:
            raise ValueError(f"Action must range in between [0, 728], got {action}")
        # Determine the row number (0-8)
        row = action // 81
        # Determine the column number (0-8)
        col = (action // 9) % 9
        # Determine the number to add (1-9)
        num = action % 9 + 1
        if self.play_grid[row, col] == 0:
            # Check if the played action is valid based on Sudoku rules
            play_grid_2 = self.play_grid.copy()
            # Add the number to the grid at the corresponding position
            play_grid_2[row, col] = num
            if sudoku_validator.is_unsolved_sudoku_valid(play_grid_2):
                # Persist the new grid
                self.play_grid = play_grid_2
                return SudokuReward.VALID_ACTION, self._is_episode_done()
            else:
                # We don't save the grid that ended up in an invalid Sudoku state
                # (grid stayed the same)
                return SudokuReward.INVALID_ACTION, False
        else:
            # Negative reward is given because there is already a number in the cell
            return SudokuReward.INVALID_ACTION, False

    def _is_episode_done(self) -> bool:
        # If there are no more 0's the game terminated
        return True if self.play_grid.all() else False

    def step(self, action: int):
        reward, terminated = self._play_action(action)
        observation = self._get_obs()
        info = self._get_info()
        return observation, reward, terminated, False, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.solved_grid, self.play_grid = sudoku_generator.generate_9x9_sudoku()
        observation = self._get_obs()
        info = self._get_info()
        return observation, info
