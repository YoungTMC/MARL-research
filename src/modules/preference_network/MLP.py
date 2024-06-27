import numpy as np
import torch.nn as nn


def _init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, std=0.01)


class MLP(nn.Module):
    def __init__(self, nums_agents, preference_agents_num, obs_dim, hidden_size):
        super(MLP, self).__init__()

        self.input_layer = nn.Linear(nums_agents * obs_dim, hidden_size)
        self.act1 = nn.ReLU()
        self.hidden_layer = nn.Linear(hidden_size, hidden_size)
        self.act2 = nn.ReLU()
        self.output_layer = nn.Linear(hidden_size, preference_agents_num * obs_dim)

        self.mlp = nn.Sequential(
            self.input_layer,
            self.act1,
            self.hidden_layer,
            self.act2,
            self.output_layer
        )

    def forward(self, x):
        return self.mlp(x)

