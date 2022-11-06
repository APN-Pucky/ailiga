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

from ailiga.Fighter import Fighter
from ailiga.TrainedFighter import TrainedFighter


class DQNFighter_v0(TrainedFighter):
    def compatible_envs(self):
        return ["tictactoe_v3"]

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
        net = Net(
            state_shape=observation_space.shape or observation_space.n,
            action_shape=env.action_space.shape or env.action_space.n,
            hidden_sizes=[128, 128, 128, 128],
            device=device,
        ).to(device)
        optim = torch.optim.Adam(net.parameters(), lr=1e-4)
        agent_learn = DQNPolicy(
            model=net,
            optim=optim,
            discount_factor=0.9,
            estimation_step=3,
            target_update_freq=320,
        )
        self.policy = agent_learn
        self.load(savefile)

    def train(self, savefile="trained/DQNFigher_v0.pth"):
        train_envs = DummyVectorEnv([self.lambda_env for _ in range(10)])
        test_envs = DummyVectorEnv([self.lambda_env for _ in range(10)])

        # seed
        seed = 1
        np.random.seed(seed)
        torch.manual_seed(seed)
        train_envs.seed(seed)
        test_envs.seed(seed)

        # ======== Step 2: Agent setup =========
        # policy, optim, agents = _get_agents()
        agents = [RandomPolicy(), self.policy]
        policy = MultiAgentPolicyManager(agents, self.env)
        agents = self.env.agents

        # ======== Step 3: Collector setup =========
        train_collector = Collector(
            policy,
            train_envs,
            VectorReplayBuffer(20_000, len(train_envs)),
            exploration_noise=True,
        )
        test_collector = Collector(policy, test_envs, exploration_noise=True)
        # policy.set_eps(1)
        train_collector.collect(n_step=64 * 10)  # batch size * training_num

        # ======== Step 4: Callback functions setup =========
        def save_best_fn(policy):
            model_save_path = os.path.join("log", "rps", "dqn", "policy.pth")
            os.makedirs(os.path.join("log", "rps", "dqn"), exist_ok=True)
            torch.save(policy.policies[agents[1]].state_dict(), model_save_path)

        def stop_fn(mean_rewards):
            return mean_rewards >= 0.6

        def train_fn(epoch, env_step):
            policy.policies[agents[1]].set_eps(0.1)

        def test_fn(epoch, env_step):
            policy.policies[agents[1]].set_eps(0.05)

        def reward_metric(rews):
            return rews[:, 1]

        result = offpolicy_trainer(
            policy=policy,
            train_collector=train_collector,
            test_collector=test_collector,
            max_epoch=50,
            step_per_epoch=1000,
            step_per_collect=50,
            episode_per_test=10,
            batch_size=64,
            train_fn=train_fn,
            test_fn=test_fn,
            stop_fn=stop_fn,
            save_best_fn=save_best_fn,
            update_per_step=0.1,
            test_in_train=False,
            reward_metric=reward_metric,
            logger=self.get_logger(),
        )
        torch.save(self.policy.state_dict(), savefile)

        # return result, policy.policies[agents[1]]
        print(f"\n==========Result==========\n{result}")
