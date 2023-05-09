"""
Shared fixtures used in test package
"""
import pytest

from sudoku_rl import utils


@pytest.fixture(scope="function", autouse=True)
def reproducible_seed():
    utils.seed_all(seed=1984)
