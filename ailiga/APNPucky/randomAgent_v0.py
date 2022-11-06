import random

from tianshou.policy import RandomPolicy

from ailiga.agent import Agent


class RandomAgent(Agent):
    def __init__(self, env):
        super().__init__(env)
        self.policy = RandomPolicy()
