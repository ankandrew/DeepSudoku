"""
Test env module
"""
import numpy as np
from gymnasium.spaces import Box, Discrete

from sudoku_rl.env import SudokuEnv, SudokuReward


def test_action_space():
    env = SudokuEnv()
    assert isinstance(env.action_space, Discrete)
    assert env.action_space.n == 9 * 9 * 9


def test_observation_space():
    env = SudokuEnv()
    assert isinstance(env.observation_space, Box)
    assert env.observation_space.shape == (9 * 9,)


def test_valid_action():
    env = SudokuEnv()
    # Pick an empty cell
    row, col = 0, 0
    while env.play_grid[row, col] != 0:
        col += 1
        if col > 8:
            row += 1
            col = 0
    action = row * 81 + col * 9 + 1
    _, reward, *_ = env.step(action)
    assert reward == SudokuReward.VALID_ACTION


def test_invalid_action():
    env = SudokuEnv()
    # Pick a cell with a number already in it
    row, col = 0, 0
    while env.play_grid[row, col] == 0:
        col += 1
        if col > 8:
            row += 1
            col = 0
    action = row * 81 + col * 9 + env.play_grid[row, col]
    _, reward, *_ = env.step(action)
    assert reward == SudokuReward.INVALID_ACTION


def test_win_episode():
    env = SudokuEnv()
    env.play_grid = env.solved_grid
    _, reward, done, *_ = env.step(0)
    assert reward == SudokuReward.WIN
    assert done


def test_reset_changes_grid():
    env = SudokuEnv()
    old_grid = env.play_grid.copy()
    env.reset()
    new_grid = env.play_grid.copy()
    with np.testing.assert_raises(AssertionError):
        np.testing.assert_array_equal(old_grid, new_grid)
