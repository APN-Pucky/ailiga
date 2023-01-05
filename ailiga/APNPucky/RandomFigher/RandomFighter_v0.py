import random

from pettingzoo.classic import rps_v2, tictactoe_v3
from tianshou.policy import RandomPolicy

from ailiga.fighter import Fighter


class RandomFighter_v0(Fighter):
    user = "APNPucky"

    @classmethod
    def compatible_envs(cls):
        return [
            "rps_v2",
            "tictactoe_v3",
            "simple_spread_v2",
            "knights_archers_zombies_v10",
        ]

    def __init__(self, env):
        super().__init__(env)
        self.policy = RandomPolicy()
