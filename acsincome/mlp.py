import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, num_features: int=10, hidden_size: int=64, num_classes: int=2):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(num_features, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_classes)
        ).to(dtype=torch.float)

    def forward(self, x):
        return self.mlp(x)
