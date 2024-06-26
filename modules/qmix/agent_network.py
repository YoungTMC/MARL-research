import torch.nn as nn


"""
QMIX RNN Agents:
    MLP + GRU(input + history) + MLP
"""


agent_REGISTRY = {}


class NRNNAgent(nn.Module):
    def __init__(self, input_dim, args):
        super(NRNNAgent, self).__init__()
        self.args = args

        # MLP + GRU + MLP
        self.mlp1 = nn.Linear(input_dim, args.rnn_hidden_dim)
        self.gru = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.mlp2 = nn.Linear(args.rnn_hidden_dim, args.n_actions)

        # init
        orthogonal_init_(self.mlp1, gain=1)
        orthogonal_init_(self.mlp2, gain=2)

    def forward(self, obs, hidden_state):
        """
        :return: each agent's q_value
        """
        x = nn.Sequential(
            self.mlp1,
            nn.ReLU()
        )
        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        h = self.gru(x, h_in)
        q = self.mlp2(h)
        print("h: {}, q:{}", h, q)
        return q, h

    def init_hidden_state(self):
        """
        :return: initialized hidden state
        """
        return self.mlp1.weight.new(1, self.args.rnn_hidden_dim).zero_()


agent_REGISTRY["n_rnn"] = NRNNAgent


def init(module, weight_init, bias_init, gain=1):
    """
    Initializing the given module by weight_init & bias_init function.

    :param module: NN module to be initialized.
    :param weight_init: function to be used to initialize module's weights.
    :param bias_init: function to be used to initialize module's biases.
    :param gain: factor adjusts the scale of the weights.
    :return: module after initialization
    """
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


def orthogonal_init_(m, gain=1):
    """
    Orthogonal initialization function, usually initializes the Linear layer,
    in order to maintain the independence between different features.
    """
    if isinstance(m, nn.Linear):
        init(m, nn.init.orthogonal_,
             lambda x: nn.init.constant_(x, 0), gain)
