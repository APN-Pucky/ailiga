import random

from pettingzoo.classic import rps_v2, tictactoe_v3
from tianshou.policy import RandomPolicy

from ailiga.Fighter import Fighter


class RandomFighter_v0(Fighter):
    def compatible_envs(self):
        return ["rps_v2", "tictactoe_v3"]

    def __init__(self, env):
        super().__init__(env)
        self.policy = RandomPolicy()
