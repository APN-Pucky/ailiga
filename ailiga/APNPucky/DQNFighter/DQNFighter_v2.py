import os
from typing import Optional, Tuple

import gym
import numpy as np
import torch
from pettingzoo.classic import rps_v2, tictactoe_v3
from tianshou.data import Collector, VectorReplayBuffer
from tianshou.env import DummyVectorEnv, PettingZooEnv
from tianshou.env.pettingzoo_env import PettingZooEnv
from tianshou.policy import BasePolicy, DQNPolicy, MultiAgentPolicyManager, RandomPolicy
from tianshou.trainer import offpolicy_trainer
from tianshou.utils import TensorboardLogger
from tianshou.utils.net.common import Net
from torch.utils.tensorboard import SummaryWriter

from ailiga.APNPucky.DQNFighter.DQNFighter_v0 import DQNFighter_v0
from ailiga.APNPucky.DQNFighter.DQNFighter_v1 import DQNFighter_v1
from ailiga.Fighter import Fighter
from ailiga.TrainedFighter import TrainedFighter


class DQNFighter_v2(DQNFighter_v1):

    reward_threshold = 0.9

    def train(self, savefile=None):
        if savefile is None:
            savefile = self.get_default_savefile()
        train_envs = DummyVectorEnv([self.lambda_env for _ in range(self.training_num)])
        test_envs = DummyVectorEnv([self.lambda_env for _ in range(self.test_num)])

        # seed
        #        seed = 1
        #        np.random.seed(seed)
        #        torch.manual_seed(seed)
        #        train_envs.seed(seed)
        #        test_envs.seed(seed)
        for layer in self.policy.children():
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()

        # ======== Step 2: Agent setup =========
        # policy, optim, agents = _get_agents()
        agents = [DQNFighter_v0(self.lambda_env).get_policy(), self.policy]
        policy = MultiAgentPolicyManager(agents, self.env)
        agents = self.env.agents

        # ======== Step 3: Collector setup =========
        train_collector = Collector(
            policy,
            train_envs,
            VectorReplayBuffer(self.buffer_size, len(train_envs)),
            exploration_noise=True,
        )
        test_collector = Collector(policy, test_envs, exploration_noise=True)
        # policy.set_eps(1)
        # batch size * training_num
        train_collector.collect(n_step=self.batch_size * self.training_num)

        # ======== Step 4: Callback functions setup =========
        def save_best_fn(policy):
            torch.save(policy.policies[agents[1]].state_dict(), savefile)

        def stop_fn(mean_rewards):
            return mean_rewards >= self.reward_threshold

        def train_fn(epoch, env_step):
            policy.policies[agents[1]].set_eps(self.train_eps)

        def test_fn(epoch, env_step):
            policy.policies[agents[1]].set_eps(self.test_eps)

        def reward_metric(rews):
            return rews[:, 1]

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
            save_best_fn=save_best_fn,
            update_per_step=self.update_per_step,
            test_in_train=False,
            reward_metric=reward_metric,
            logger=self.get_logger(),
        )
        torch.save(self.policy.state_dict(), savefile)

        # return result, policy.policies[agents[1]]
        print(f"\n==========Result==========\n{result}")
