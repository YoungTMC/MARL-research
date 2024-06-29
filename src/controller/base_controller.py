import torch

from tools.action_selectors.action_selector import action_selector_REGISTRY
from modules.agents import agent_REGISTRY


def get_parameters_num(param_list):
    return str(sum(p.numel() for p in param_list) / 1000) + 'K'


class BaseMultiAgentController(object):
    def __init__(self, obs_scheme, groups, args):
        self.args = args
        self.n_agents = args.n_agents
        self.agent_output_type = args.agent_output_type
        self.input_shape = self._get_input_shape(obs_scheme)
        self._build_agents(self.input_shape)
        self.action_selector = action_selector_REGISTRY[self.args.action_selector](args)

        self.hidden_states = None

    def select_actions(self, ep_batch, t_ep, t_env, batch_slice=slice(None), test_mode=False):
        """
        :param ep_batch: batch with multiple episodes
        :param t_ep: data index of ep_batch
        :param t_env: environment's timestep
        :param batch_slice: e.g. slice(5, 10) - choose batch data with index from 5 to 9;
        None means choose the whole batch.
        :param test_mode:
        :return:
        """
        if t_ep == 0:
            self.set_evaluation_mode()

        avail_actions = ep_batch["avail_actions"][:, t_ep]
        q_values = self.forward(ep_batch, t_ep, test_mode=test_mode)
        chosen_actions = self.action_selector.choose_action(
            q_values[batch_slice], avail_actions[batch_slice], t_env, test_mode=test_mode)

        return chosen_actions

    def forward(self, ep_batch, t, test_mode=False):
        agent_inputs = self._build_inputs(ep_batch, t)
        avail_actions = ep_batch["avail_actions"][:, t]
        q_values, self.hidden_states = self.agent(agent_inputs, self.hidden_states)

        # Softmax the agent outputs if they're policy logits
        if self.agent_output_type == "pi_logits":
            if getattr(self.args, "mask_before_softmax", True):
                # Make the logits for unavailable actions very negative to minimise their affect on the softmax
                q_values = q_values.reshape(ep_batch.batch_size * self.n_agents, -1)
                reshaped_avail_actions = avail_actions.reshape(ep_batch.batch_size * self.n_agents, -1)
                q_values[reshaped_avail_actions == 0] = -1e10

            q_values = torch.nn.functional.softmax(q_values, dim=-1)

        return q_values.view(ep_batch.batch_size, self.n_agents, -1)

    def init_hidden_state(self, batch_size):
        self.hidden_states = self.agent.init_hidden()
        if self.hidden_states is not None:
            self.hidden_states = self.hidden_states.unsqueeze(0).expand(batch_size, self.n_agents, -1)

    def set_train_mode(self):
        self.agent.train()

    def set_evaluation_mode(self):
        self.agent.eval()

    def load_state(self, other_controller):
        self.agent.load_state_dict(other_controller.agent.state_dict())

    def cuda(self):
        self.agent.cuda()

    def cpu(self):
        self.agent.cpu()

    def get_device(self):
        return next(self.parameters()).device

    def save_models(self, path):
        torch.save(self.agent.state_dict(), "{}/agent.th".format(path))

    def load_models(self, path):
        self.agent.load_state_dict(torch.load("{}/agent.th".format(path), map_location=lambda storage, loc: storage))

    def _get_input_shape(self, obs_scheme):
        input_shape = obs_scheme["obs"]["vshape"]
        if self.args.obs_last_action:
            input_shape += obs_scheme["actions_onehot"]["vshape"][0]
        if self.args.obs_agent_id:
            input_shape += self.n_agents

        return input_shape

    def _build_agents(self, input_shape):
        self.agent = agent_REGISTRY[self.args.agent](input_shape, self.args)
        print("&&&&&&&&&&&&&&&&&&&&&&", self.args.agent, get_parameters_num(self.parameters()))

    def _build_inputs(self, batch, t):
        # Assumes homogenous agents with flat observations.
        # Other MACs might want to e.g. delegate building inputs to each agent
        bs = batch.batch_size
        inputs = []
        inputs.append(batch["obs"][:, t])  # b1av
        if self.args.obs_last_action:
            if t == 0:
                inputs.append(torch.zeros_like(batch["actions_onehot"][:, t]))
            else:
                inputs.append(batch["actions_onehot"][:, t-1])
        if self.args.obs_agent_id:
            inputs.append(torch.eye(self.n_agents, device=batch.device).unsqueeze(0).expand(bs, -1, -1))

        inputs = torch.cat([x.reshape(bs, self.n_agents, -1) for x in inputs], dim=-1)
        return inputs

    def parameters(self):
        return self.agent.parameters()
