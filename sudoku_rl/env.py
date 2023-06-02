"""
Check: https://www.gymlibrary.dev/content/environment_creation/
"""

import random
from dataclasses import astuple, dataclass
from enum import IntEnum, auto

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from sudoku_rl import sudoku_generator, sudoku_validator, utils

WIN_REWARD: float = 1
"""Ultimate reward when agent fills all the Sudoku cells and it's a valid grid."""
VALID_ACTION_REWARD: float = 0.1
"""Reward given when the agent plays a valid number in a playable cell."""
NO_REWARD: float = 0.0
"""No Reward"""
INVALID_ACTION_REWARD: float = -1
"""Negative reward given to the agent when tries to fill in cells that were originally filled in."""


@dataclass
class Position:
    """Position"""

    row: int
    col: int

    def move_left(self) -> None:
        self.col = max(0, self.col - 1)

    def move_right(self) -> None:
        self.col = min(8, self.col + 1)

    def move_up(self) -> None:
        self.row = max(0, self.row - 1)

    def move_down(self) -> None:
        self.row = min(8, self.row + 1)

    def __post_init__(self) -> None:
        assert 0 <= self.col <= 8, f"Invalid {self.col} value for col"
        assert 0 <= self.row <= 8, f"Invalid {self.row} value for row"


@dataclass
class ActionResult:
    """Action Result"""

    reward: float
    terminated: bool

    def __iter__(self):
        return iter(astuple(self))


class Action(IntEnum):
    """Action valid options"""

    MOVE_LEFT = 0
    MOVE_RIGHT = auto()
    MOVE_UP = auto()
    MOVE_DOWN = auto()
    INSERT_1 = auto()
    INSERT_2 = auto()
    INSERT_3 = auto()
    INSERT_4 = auto()
    INSERT_5 = auto()
    INSERT_6 = auto()
    INSERT_7 = auto()
    INSERT_8 = auto()
    INSERT_9 = auto()


class SudokuEnv(gym.Env):
    """Custom Sudoku Environment that follows gym interface."""

    metadata = {"render.modes": ["human"]}

    def __init__(self):
        super().__init__()
        # We can: go up, down, left, right (4 actions) + place any number from [1, 9] (9 actions)
        self.action_space = spaces.Discrete(4 + 9)
        # In each cell of the 9x9 grid we can observe a value from [0, 9] where 0 indicates
        # that a value needs to be filled
        self.observation_space = spaces.MultiBinary(9 * 9 * 10)
        self._new_sudoku()
        self.position = Position(row=random.randint(0, 8), col=random.randint(0, 8))

    def _new_sudoku(self) -> None:
        self.solved_grid, self.play_grid = sudoku_generator.generate_9x9_sudoku()
        self.solved_grid.flags.writeable = False

    def _get_info(self):
        return {"filled_cells": np.count_nonzero(self.play_grid)}

    def _get_obs(self):
        return utils.one_hot_9x9_sudoku(self.play_grid).ravel()

    def _play_action(self, action: int) -> ActionResult:
        if not 0 <= action <= 12:
            raise ValueError(f"Action must range in between [0, 12], got {action}")

        if action == Action.MOVE_LEFT:
            self.position.move_left()
            return ActionResult(reward=NO_REWARD, terminated=False)
        if action == Action.MOVE_RIGHT:
            self.position.move_right()
            return ActionResult(reward=NO_REWARD, terminated=False)
        if action == Action.MOVE_UP:
            self.position.move_up()
            return ActionResult(reward=NO_REWARD, terminated=False)
        if action == Action.MOVE_DOWN:
            self.position.move_down()
            return ActionResult(reward=NO_REWARD, terminated=False)

        if action == Action.INSERT_1:
            play_number = 1
        elif action == Action.INSERT_2:
            play_number = 2
        elif action == Action.INSERT_3:
            play_number = 3
        elif action == Action.INSERT_4:
            play_number = 4
        elif action == Action.INSERT_5:
            play_number = 5
        elif action == Action.INSERT_6:
            play_number = 6
        elif action == Action.INSERT_7:
            play_number = 7
        elif action == Action.INSERT_8:
            play_number = 8
        elif action == Action.INSERT_9:
            play_number = 9

        if self.play_grid[self.position.row, self.position.col] == 0:
            # Check if the played action is valid based on Sudoku rules
            play_grid_2 = self.play_grid.copy()
            # Add the number to the grid at the corresponding position
            play_grid_2[self.position.row, self.position.col] = play_number
            if sudoku_validator.is_sudoku_valid(play_grid_2):
                if self.is_episode_done():
                    return ActionResult(reward=WIN_REWARD, terminated=True)
                else:
                    # Game continues, persist the new grid
                    self.play_grid = play_grid_2
                    return ActionResult(reward=VALID_ACTION_REWARD, terminated=False)
            else:
                # We ended up in an invalid Sudoku state
                return ActionResult(reward=INVALID_ACTION_REWARD, terminated=False)
        else:
            # There is already a number in the cell
            return ActionResult(reward=NO_REWARD, terminated=False)

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
