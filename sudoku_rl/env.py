"""
Check: https://www.gymlibrary.dev/content/environment_creation/
"""

from typing import Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from sudoku_rl import sudoku_generator, sudoku_validator, utils

WIN_REWARD: float = 1.0
"""Ultimate reward when agent fills all the Sudoku cells and it's a valid grid."""
VALID_ACTION_REWARD: float = 0.1
"""Reward given when the agent plays a valid number in a playable cell."""
INVALID_ACTION_REWARD: float = -0.01
"""Negative reward given to the agent when tries to fill in cells that were originally filled in."""


class SudokuEnv(gym.Env):
    """Custom Sudoku Environment that follows gym interface."""

    metadata = {"render.modes": ["human"]}

    def __init__(self):
        super().__init__()
        # In each cell of the 9x9 grid we can put a value from [1, 9], hence 9**3
        self.action_space = spaces.Discrete(9 * 9 * 9)
        # In each cell of the 9x9 grid we can observe a value from [0, 9] where 0 indicates
        # that a value needs to be filled
        self.observation_space = spaces.MultiBinary(9 * 9 * 10)
        self._new_sudoku()

    def _new_sudoku(self) -> None:
        self.solved_grid, self.play_grid = sudoku_generator.generate_9x9_sudoku()
        self.solved_grid.flags.writeable = False

    def _get_info(self):
        return {"filled_cells": np.count_nonzero(self.play_grid)}

    def _get_obs(self):
        return utils.one_hot_9x9_sudoku(self.play_grid).ravel()

    def _play_action(self, action: int) -> Tuple[float, bool]:
        if not 0 <= action <= 728:
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
            if sudoku_validator.is_sudoku_valid(play_grid_2):
                # Game continues, persist the new grid
                self.play_grid = play_grid_2
                if self.is_episode_done():
                    return WIN_REWARD, True
                else:
                    return VALID_ACTION_REWARD, False
            else:
                # We ended up in an invalid Sudoku state
                return INVALID_ACTION_REWARD, True
        else:
            # There is already a number in the cell
            return INVALID_ACTION_REWARD, True

    def is_episode_done(self) -> bool:
        # If there are no more 0's the game terminated
        return np.count_nonzero(self.play_grid == 0) == 0

    def step(self, action: int):
        reward, terminated = self._play_action(action)
        observation = self._get_obs()
        info = self._get_info()
        return observation, reward, terminated, False, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._new_sudoku()
        observation = self._get_obs()
        info = self._get_info()
        return observation, info
