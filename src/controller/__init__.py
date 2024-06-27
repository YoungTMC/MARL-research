REGISTRY = {}

from .base_controller import BaseMultiAgentController
REGISTRY["base_controller"] = BaseMultiAgentController

from .n_controller import NMultiAgentController
REGISTRY["n_controller"] = NMultiAgentController
