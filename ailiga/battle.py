import multiprocessing as mp

import numpy as np
import tqdm
from tianshou.data import Collector
from tianshou.env import DummyVectorEnv, RayVectorEnv, SubprocVectorEnv
from tianshou.policy import BasePolicy, DQNPolicy, MultiAgentPolicyManager, RandomPolicy

from ailiga import env as menv


class Battle:
    """Runs a battle between two or more agents."""

    def __init__(self, lambda_env, agents):
        self.env = lambda_env()
        self.lambda_env = lambda_env
        if isinstance(agents[0], type):
            # agents are classes, not instances
            self.agents = [a(self.lambda_env) for a in agents]
        else:
            self.agents = agents
        self.policies = [a.get_policy() for a in self.agents]
        self.env.reset()
        self.rews = None
        self.lens = None
        if len(self.env.agents) != len(self.agents):
            raise ValueError(
                "Agents do not match environment: "
                + str(self.env.agents)
                + " vs "
                + str(self.agents)
            )

    def fight(self, n_episodes=1, n_step=None, render=None, n_jobs=None):
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
            # DummyVectorEnv([lambda: env for _ in range(1)]),
            # SubprocVectorEnv([lambda: env for _ in range(10)]),
            SubprocVectorEnv(
                [
                    lambda: env
                    for _ in range(n_jobs if n_jobs is not None else mp.cpu_count())
                ]
            ),
            exploration_noise=True,
        )
        result = collector.collect(n_episode=n_episodes, n_step=n_step, render=render)
        self.rews, self.lens = result["rews"], result["lens"]
        return [self.rews[:, i].mean() for i in range(len(self.agents))]
