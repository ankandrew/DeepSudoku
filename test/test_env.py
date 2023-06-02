"""
Test env module
"""
import pytest
from gymnasium.spaces import Discrete, MultiBinary
from stable_baselines3.common import env_checker

from sudoku_rl.env import SudokuEnv


@pytest.fixture(scope="function")
def env() -> SudokuEnv:
    return SudokuEnv()


def test_action_space(env: SudokuEnv):
    assert isinstance(env.action_space, Discrete)
    assert env.action_space.n == 4 + 9


def test_observation_space(env: SudokuEnv):
    assert isinstance(env.observation_space, MultiBinary)
    assert env.observation_space.shape == (9 * 9 * 10,)


def test_env_is_valid(env: SudokuEnv) -> None:
    env_checker.check_env(env)
