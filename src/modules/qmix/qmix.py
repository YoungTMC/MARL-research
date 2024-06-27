import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class QMIX(nn.Module):
    def __init__(self, args):
        super(QMIX, self).__init__()
        self.args = args
        self.state_dim = int(np.prod(args.state_shape))

        if self.args.two_hyper_layers:
            self.hyper_w1 = nn.Sequential(
                nn.Linear(self.state_dim, self.args.hyper_hidden_dim),
                nn.ReLU(),
                nn.Linear(self.args.hyper_hidden_dim, self.args.n_agents * self.args.qmix_hidden_dim)
            )
            self.hyper_w2 = nn.Sequential(
                nn.Linear(self.state_dim, self.args.hyper_hidden_dim),
                nn.ReLU(),
                nn.Linear(self.args.hyper_hidden_dim, self.args.qmix_hidden_dim * 1)
            )
        else:
            self.hyper_w1 = nn.Linear(self.state_dim, self.args.n_agents * self.args.qmix_hidden_dim)
            self.hyper_w2 = nn.Linear(self.state_dim, self.args.qmix_hidden_dim * 1)

        self.hyper_b1 = nn.Linear(self.state_dim, self.args.qmix_hidden_dim)
        self.hyper_b2 = nn.Sequential(
            nn.Linear(self.state_dim, self.args.hyper_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.args.hyper_hidden_dim, 1)
        )

    def forward(self, q_values, states):
        """
        Architecture of Mixing Network.
        :param q_values: output of agent network
        :param states: states of environment
        :return Q_total
        """
        episode_num = q_values.size(0)
        q_values = q_values.view(-1, 1, self.args.n_agents)
        states = states.reshape(-1, self.state_dim)

        w1 = torch.abs(self.hyper_w1(states))
        b1 = self.hyper_b1(states)
        # [episode_num * 1 * n_agents] * [episode_num * n_agents * qmix_hidden_dim] =
        # [episode_num * 1 * qmix_hidden_dim]
        w1 = w1.view(-1, self.args.n_agents, self.args.qmix_hidden_dim)
        b1 = b1.view(-1, 1, self.args.qmix_hidden_dim)
        hidden = F.elu(torch.bmm(q_values, w1), b1)

        w2 = torch.abs(self.hyper_w2(states))
        b2 = self.hyper_b2(states)
        # [episode_num * 1 * qmix_hidden_dim] * [episode_num * qmix_hidden_dim * 1] =
        # [episode_num * 1 * 1]
        w2 = w2.view(-1, self.args.qmix_hidden_dim, 1)
        b2 = b2.view(-1, 1, 1)
        q_total = torch.bmm(hidden, w2) + b2
        q_total = q_total.view(episode_num, -1, 1)
        return q_total
