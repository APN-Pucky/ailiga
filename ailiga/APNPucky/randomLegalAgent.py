import numpy as np

from ailiga.agent import Agent


class RandomLegalAgent(Agent):
    def __init__(self, env):
        Agent.__init__()
        self.num_actions = env.action_space(env.possible_agents[0]).n

    def act(self, observation):
        a = np.array(range(self.num_actions))[observation["action_mask"] == 1]
        r = a[np.random.randint(0, len(a))]
        # print(np.array(range(self.num_actions)),observation['action_mask'] ,a ," -> ", r)
        return r

    def episode_over(self, reward):
        pass

    def reset(self):
        pass
