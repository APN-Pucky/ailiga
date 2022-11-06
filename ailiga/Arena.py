import numpy as np
import tqdm
from tianshou.data import Collector
from tianshou.env import DummyVectorEnv, SubprocVectorEnv
from tianshou.policy import BasePolicy, DQNPolicy, MultiAgentPolicyManager, RandomPolicy


class Battle:
    def __init__(self, lambda_env, agents):
        self.env = lambda_env()
        self.agents = agents
        self.policies = [a.get_policy() for a in self.agents]
        self.env.reset()

    def fight(self, n_episodes=1, n_step=None, render=None):
        """
        Runs a number of episodes between two agents.

        :param n_episodes: number of episodes to run
        :param n_step: number of steps per episode
        :param render: if True, render the environment
        :return: list of rewards
        """
        env = self.env
        policy = MultiAgentPolicyManager(self.policies, self.env)
        policy.eval()
        # policy.policies[agents[args.agent_id - 1]].set_eps(0.05)
        collector = Collector(
            policy,
            SubprocVectorEnv([lambda: env for _ in range(10)]),
            exploration_noise=True,
        )
        result = collector.collect(n_episode=n_episodes, n_step=n_step, render=render)
        rews, lens = result["rews"], result["lens"]
        return [rews[:, i].mean() for i in range(len(self.agents))]


class Tournament:
    def __init__(self, lambda_env, agents):
        self.lambda_env = lambda_env
        self.agents = agents

    def fight(self, n_episodes=1, n_step=None):
        """
        Battle between all fighters. Everyone fights everyone.

        :param n_episodes: number of episodes to run
        :param n_step: number of steps per episode
        :return: list of rewards
        """
        attacker = np.zeros((len(self.agents), len(self.agents)))
        defender = np.zeros((len(self.agents), len(self.agents)))
        for i in tqdm.tqdm(range(len(self.agents))):
            for j in range(len(self.agents)):
                battle = Battle(self.lambda_env, [self.agents[i], self.agents[j]])
                rews = battle.fight(n_episodes, n_step)
                attacker[i][j] = rews[0]
                defender[i][j] = rews[1]
        return attacker, defender
