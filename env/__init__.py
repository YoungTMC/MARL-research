from functools import partial
import sys
import os

from .multiagentenv import MultiAgentEnv

try:
    smac = True
    from .smac_v1 import StarCraft2EnvWrapper
except Exception as e:
    print(e)
    smac = False


def env_fn(env, **kwargs) -> MultiAgentEnv:
    return env(**kwargs)


REGISTRY = {}

if smac:
    REGISTRY["sc2"] = partial(env_fn, env=StarCraft2EnvWrapper)
    if sys.platform == "linux":
        os.environ.setdefault("SC2PATH",
                              os.path.join(os.getcwd(), "3rdparty", "StarCraftII"))
else:
    print("SMAC V1 is not supported...")


print("Supported environments:", REGISTRY)
