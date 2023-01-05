import gym
from pettingzoo.butterfly import (
    knights_archers_zombies_v10 as _knights_archers_zombies_v10,
)
from pettingzoo.classic import leduc_holdem_v4 as _leduc_holdem_v4
from pettingzoo.classic import rps_v2 as _rps_v2
from pettingzoo.classic import tictactoe_v3 as _tictactoe_v3
from pettingzoo.mpe import simple_spread_v2 as _simple_spread_v2
from tianshou.env.pettingzoo_env import PettingZooEnv


# TODO handle extra environment variables/types
def get_env_name(env):
    """Get the name of the env."""
    if callable(env):
        env = env()
    if isinstance(env, gym.Env):
        return env.spec.id
    elif isinstance(env, PettingZooEnv):
        return env.env.metadata["name"]
    else:
        raise Exception("Unknown env type")


def tictactoe_v3():
    return PettingZooEnv(_tictactoe_v3.env())


def simple_spread_v2():
    return PettingZooEnv(_simple_spread_v2.env())


def knights_archers_zombies_v10():
    return PettingZooEnv(_knights_archers_zombies_v10.env())


def leduc_holdem_v4():
    return PettingZooEnv(_leduc_holdem_v4.env())


def rps_v2():
    return PettingZooEnv(_rps_v2.env())


def get_all_envs():
    return [
        knights_archers_zombies_v10,
        tictactoe_v3,
        simple_spread_v2,
        leduc_holdem_v4,
        rps_v2,
    ]


def get_all_env_names():
    return get_envs().keys()


def get_envs():
    ret = {}
    for env in get_all_envs():
        ret[env.__name__] = env
    return ret


def get_env(name):
    """Get the env by name."""
    return get_envs()[name]
