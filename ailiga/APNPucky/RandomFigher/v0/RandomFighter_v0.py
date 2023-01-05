import random

from pettingzoo.classic import rps_v2, tictactoe_v3
from tianshou.policy import RandomPolicy

from ailiga.env import get_all_env_names
from ailiga.fighter import Fighter


class RandomFighter_v0(Fighter):
    user = "APNPucky"

    @classmethod
    def compatible_envs(cls):
        return get_all_env_names()

    def __init__(self, env):
        super().__init__(env)
        self.policy = RandomPolicy()
