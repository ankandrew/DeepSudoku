import gymnasium as gym
import numpy as np
from gymnasium import spaces


class CustomEnv(gym.Env):
    """Custom Environment that follows gym interface."""

    metadata = {"render.modes": ["human"]}

    def __init__(self, arg1, arg2):
        super().__init__()
        self.action_space = spaces.Discrete(9 * 9 * 9)
        self.observation_space = spaces.Box(low=0, high=9, shape=(9, 9), dtype=np.uint8)

    def step(self, action):
        ...
        return observation, reward, done, info

    def reset(self):
        ...
        return observation  # reward, done, info can't be included

    def render(self):
        ...

    def close(self):
        ...
