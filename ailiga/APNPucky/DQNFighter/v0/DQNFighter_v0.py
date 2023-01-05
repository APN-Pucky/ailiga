import os
from typing import Optional, Tuple

import gym
import numpy as np
import torch
from tianshou.data import Collector, VectorReplayBuffer
from tianshou.env import DummyVectorEnv, PettingZooEnv
from tianshou.env.pettingzoo_env import PettingZooEnv
from tianshou.policy import BasePolicy, DQNPolicy, MultiAgentPolicyManager, RandomPolicy
from tianshou.trainer import offpolicy_trainer
from tianshou.utils import TensorboardLogger
from tianshou.utils.net.common import Net
from torch.utils.tensorboard import SummaryWriter

from ailiga.APNPucky.RandomFigher.v0.RandomFighter_v0 import RandomFighter_v0
from ailiga.fighter import Fighter
from ailiga.trained_fighter import TrainedFighter


class DQNFighter_v0(TrainedFighter):
    user = "APNPucky"

    # https://tianshou.readthedocs.io/en/master/tutorials/dqn.html
    max_epoch = 50
    step_per_epoch = 1000
    step_per_collect = 50
    episode_per_test = 10
    batch_size = 64
    update_per_step = 0.1

    buffer_size = 20_000
    hidden_sizes = [128, 128, 128, 128]

    reward_threshold = 100

    training_num = 10
    test_num = 10

    lr = 1e-4

    train_eps = 0.1
    test_eps = 0.05
    gamma = 0.9
    n_step = 3
    target_update_freq = 320

    @classmethod
    def compatible_envs(cls):
        return [
            "tictactoe_v3",
            "simple_spread_v2",
            "knights_archers_zombies_v10",
            "leduc_holdem_v4",
        ]

    def __init__(self, lambda_env, savefile=None):
        super().__init__(lambda_env)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.env = lambda_env()
        env = self.env
        observation_space = (
            env.observation_space["observation"]
            if isinstance(env.observation_space, gym.spaces.Dict)
            else env.observation_space
        )
        print(observation_space, observation_space.shape or observation_space.n)
        print(env.action_space.shape or env.action_space.n)
        net = Net(
            state_shape=observation_space.shape or observation_space.n,
            action_shape=env.action_space.shape or env.action_space.n,
            hidden_sizes=self.hidden_sizes,
            device=device,
        ).to(device)
        optim = torch.optim.Adam(net.parameters(), lr=self.lr)
        agent_learn = DQNPolicy(
            model=net,
            optim=optim,
            discount_factor=self.gamma,
            estimation_step=self.n_step,
            target_update_freq=self.target_update_freq,
        )
        self.policy = agent_learn
        self.load(savefile)

    def train(self, savefile=None, seed=None, reset=True):
        super().train(seed, reset)

        # ======== Step 2: Agent setup =========
        # policy, optim, agents = _get_agents()
        agents = [
            *[
                RandomFighter_v0(self.lambda_env).get_policy()
                for _ in range(len(self.env.agents) - 1)
            ],
            self.policy,
        ]
        self.agentindex = -1
        policy = MultiAgentPolicyManager(agents, self.env)
        agents = self.env.agents
        # self.agent = agents[-1]

        # ======== Step 3: Collector setup =========
        train_collector = Collector(
            policy,
            self.train_envs,
            VectorReplayBuffer(self.buffer_size, len(self.train_envs)),
            exploration_noise=True,
        )
        test_collector = Collector(policy, self.test_envs, exploration_noise=True)
        # policy.set_eps(1)
        # batch size * training_num
        train_collector.collect(n_step=self.batch_size * self.training_num)

        # ======== Step 4: Callback functions setup =========

        def stop_fn(mean_rewards):
            return mean_rewards >= self.reward_threshold

        def train_fn(epoch, env_step):
            policy.policies[self.env.agents[self.agentindex]].set_eps(self.train_eps)

        def test_fn(epoch, env_step):
            policy.policies[self.env.agents[self.agentindex]].set_eps(self.test_eps)

        result = offpolicy_trainer(
            policy=policy,
            train_collector=train_collector,
            test_collector=test_collector,
            max_epoch=self.max_epoch,
            step_per_epoch=self.step_per_epoch,
            step_per_collect=self.step_per_collect,
            episode_per_test=self.episode_per_test,
            batch_size=self.batch_size,
            train_fn=train_fn,
            test_fn=test_fn,
            stop_fn=stop_fn,
            save_best_fn=self.save,
            update_per_step=self.update_per_step,
            test_in_train=False,
            reward_metric=self.reward_metric,
            logger=self.get_logger(),
        )
        # return result, policy.policies[agents[1]]
        print(f"\n==========Result==========\n{result}")
