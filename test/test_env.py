"""
Test env module
"""
import numpy as np
import pytest
from gymnasium.spaces import Box, Discrete
from stable_baselines3.common import env_checker

from sudoku_rl.env import INVALID_ACTION_REWARD, SudokuEnv


@pytest.fixture(scope="function")
def env() -> SudokuEnv:
    return SudokuEnv()


def test_action_space(env: SudokuEnv):
    assert isinstance(env.action_space, Discrete)
    assert env.action_space.n == 9 * 9 * 9


def test_observation_space(env: SudokuEnv):
    assert isinstance(env.observation_space, Box)
    assert env.observation_space.shape == (9 * 9,)


def test_invalid_action(env: SudokuEnv):
    # Pick a cell with a number already in it
    row, col = 0, 0
    while env.play_grid[row, col] == 0:
        col += 1
        if col > 8:
            row += 1
            col = 0
    action = row * 81 + col * 9 + env.play_grid[row, col]
    _, reward, *_ = env.step(action)
    assert reward == INVALID_ACTION_REWARD


def test_reset_changes_grid(env: SudokuEnv):
    old_grid = env.play_grid.copy()
    env.reset()
    new_grid = env.play_grid.copy()
    with np.testing.assert_raises(AssertionError):
        np.testing.assert_array_equal(old_grid, new_grid)


def test_env_is_valid(env: SudokuEnv) -> None:
    env_checker.check_env(env)
