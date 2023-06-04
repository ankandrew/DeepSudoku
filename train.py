"""
Train module
"""
import torch.nn as nn
from stable_baselines3 import PPO

from sudoku_rl.env import SudokuEnv

TOTAL_TIMESTEPS = 40_000_000
policy_kwargs = dict(activation_fn=nn.ReLU, net_arch=[128, 128])
env = SudokuEnv()

# Define and Train the agent
model = PPO(
    "MlpPolicy",
    env,
    policy_kwargs=policy_kwargs,
    verbose=2,
    tensorboard_log="tb-log/",
)
experiment_name = f"sudoku-{model.__class__.__name__}-{TOTAL_TIMESTEPS / 1e6:.2f}m-steps"
model.learn(total_timesteps=TOTAL_TIMESTEPS, tb_log_name=experiment_name)
model.save(f"trained-models/{experiment_name}")
