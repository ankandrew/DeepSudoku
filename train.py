"""
Train module
"""
from stable_baselines3 import PPO

from sudoku_rl.env import SudokuEnv

TOTAL_TIMESTEPS = 100_000

env = SudokuEnv()
# Define and Train the agent
model = PPO("MlpPolicy", env, verbose=2)
model.learn(total_timesteps=TOTAL_TIMESTEPS)
model.save(f"trained-models/sudoku-{model.__class__.__name__}-{TOTAL_TIMESTEPS / 1e6:.2f}m-steps")
