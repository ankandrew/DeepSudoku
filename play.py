"""
Play module
"""
from stable_baselines3 import PPO

from sudoku_rl.env import SudokuEnv
from sudoku_rl.utils import seed_all

env = SudokuEnv()
model = PPO.load("trained-models/sudoku-PPO-5.00m-steps", env)

seed_all(12)
env.play_grid = env.solved_grid.copy()
env.play_grid[0, 0] = 0
obs = env._get_obs()
for i in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, dones, _, info = env.step(action)
    print(f"rewards={rewards}, dones={dones}, info={info}")
    print(f"Grid:\n{env.play_grid}")
    pass
