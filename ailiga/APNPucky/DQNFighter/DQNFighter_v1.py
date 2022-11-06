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
from ailiga.Fighter import Fighter
from ailiga.TrainedFighter import TrainedFighter


class DQNFighter_v1(DQNFighter_v0):

    # https://tianshou.readthedocs.io/en/master/tutorials/dqn.html
    max_epoch = 100
    step_per_epoch = 1000
    step_per_collect = 500
    episode_per_test = 10
    batch_size = 64
    update_per_step = 0.1

    buffer_size = 20_000
    hidden_sizes = [128, 128, 128, 128]

    reward_threshold = 0.9

    training_num = 100
    test_num = 20

    lr = 1e-4

    train_eps = 0.1
    test_eps = 0.05
    gamma = 0.9
    n_step = 3
    target_update_freq = 320
