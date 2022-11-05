import random

from ailiga.agent import Agent


class RandomAgent(Agent):
    def __init__(self, env):
        Agent.__init__()
        self.num_actions = env.action_space(env.possible_agents[0]).n

    def act(self, observation):
        return random.randint(0, self.num_actions - 1)

    def episode_over(self, reward):
        pass

    def reset(self):
        pass
