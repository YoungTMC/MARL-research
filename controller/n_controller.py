from .base_controller import BaseMultiAgentController


class NMultiAgentController(BaseMultiAgentController):
    def __init__(self, scheme, args):
        super(NMultiAgentController, self).__init__(scheme, args)

    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):
        if t_ep == 0:
            self.set_evaluation_mode()
        # Only select actions for the selected batch elements in bs
        avail_actions = ep_batch["avail_actions"][:, t_ep]
        q_values = self.forward(ep_batch, t_ep, test_mode=test_mode)
        chosen_actions = self.action_selector.select_action(q_values[bs], avail_actions[bs], t_env, test_mode=test_mode)
        return chosen_actions

    def forward(self, ep_batch, t, test_mode=False):
        agent_inputs = self._build_inputs(ep_batch, t)
        q_values, self.hidden_states = self.agent(agent_inputs, self.hidden_states)
        return q_values
