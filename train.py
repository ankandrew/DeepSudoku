"""
Train module
"""
import torch.nn as nn
import torch
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from sudoku_rl.env import SudokuEnv


class MinTransformerEncoder(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(
        self,
        observation_space: spaces.Box,
        features_dim: int = 64,
        n_embd: int = 8,
        dropout: float = 0.01,
    ):
        super().__init__(observation_space, features_dim)

        self.wte = nn.Embedding(1 + 9, n_embd)
        self.wpe = nn.Embedding(9 * 9, n_embd)
        self.drop = nn.Dropout(dropout)
        self.transformer = nn.Sequential(
            nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=n_embd, nhead=2, dim_feedforward=n_embd, batch_first=True
                ),
                num_layers=2,
            ),
            nn.ReLU(),
            nn.Flatten(),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        obs = obs.long().flatten(1, 2)
        return self.transformer(self.drop(self.wte(obs) + self.wpe(obs)))


policy_kwargs = dict(
    # Feature Extractor
    features_extractor_class=MinTransformerEncoder,
    features_extractor_kwargs=dict(features_dim=648),
    # FCN
    activation_fn=nn.ReLU,
    net_arch=[],
)

TOTAL_TIMESTEPS = 10_000_000
env = SudokuEnv()

# Define and Train the agent
model = PPO(
    "MlpPolicy",
    env,
    policy_kwargs=policy_kwargs,
    # learning_rate=1e-3,
    verbose=2,
    # n_steps=64,
    # batch_size=16,
    # gamma=0.98,
    tensorboard_log="tb-log/",
)
experiment_name = f"sudoku-{model.__class__.__name__}-{TOTAL_TIMESTEPS / 1e6:.2f}m-steps"
model.learn(total_timesteps=TOTAL_TIMESTEPS, tb_log_name=experiment_name)
model.save(f"trained-models/{experiment_name}")
