import torch
from torch.distributions import Categorical
from .episilon_scheduler import DecayThenFlatScheduler


qmix_selector_REGISTRY = {}


class EpsilonGreedyActionSelector:
    def __init__(self, args):
        self.args = args

        self.scheduler = DecayThenFlatScheduler(self.args.epsilon_start, self.args.epsilon_finish,
                                                self.args.epsilon_anneal_time, decay="linear")

        self.epsilon = self.scheduler.eval(0)

    def choose_action(self, q_values, avail_actions, t_env, test_mode=False):
        self.epsilon = self.scheduler.eval(t_env)

        # greedy action selection only
        if test_mode:
            self.epsilon = getattr(self.args, "test_noise", 0.0)

        # masked actions that are unavailable
        masked_q_values = q_values.clone()
        masked_q_values[avail_actions == 0] = - float("inf")

        random_numbers = (torch.rand(size=q_values[:, :, 0].size(), dtype=torch.float32, device="cpu")
                          .to(q_values.device))
        picked_numbers = (random_numbers < self.epsilon).long()
        random_actions = Categorical(avail_actions.cpu().float()).sample().long().to(avail_actions.device)
        picked_actions = picked_numbers * random_actions + (1 - picked_numbers) * masked_q_values.max(dim=2)[1]
        return picked_actions


qmix_selector_REGISTRY["epsilon_greedy"] = EpsilonGreedyActionSelector
