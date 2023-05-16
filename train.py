"""
Train module
"""
from stable_baselines3 import PPO

from sudoku_rl.env import SudokuEnv

TOTAL_TIMESTEPS = 1_000_000

env = SudokuEnv()
# Define and Train the agent
model = PPO(
    policy="MlpPolicy",
    env=env,
    n_steps=128,
    batch_size=32,
    n_epochs=5,
    gamma=0.999,
    gae_lambda=0.98,
    ent_coef=0.01,
    verbose=1)
model.learn(total_timesteps=TOTAL_TIMESTEPS)
model.save(f"trained-models/sudoku-ppo-{TOTAL_TIMESTEPS / 1e6:.2f}m-steps")
