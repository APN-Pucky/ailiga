import itertools
import warnings
from typing import Any, Dict, Optional, Union

import gym
import numpy as np
import torch
from tianshou.data import Batch
from tianshou.policy import BasePolicy
from tianshou.utils import MultipleLRSchedulers

from ailiga.APNPucky.IndexFighter.v0.IndexFighter_v0 import IndexPolicy
from ailiga.env import get_all_env_names
from ailiga.fighter import Fighter


class RandomIndexFighter_v0(Fighter):
    user = "APNPucky"

    @classmethod
    def compatible_envs(cls):
        return get_all_env_names()

    def __init__(self, env):
        super().__init__(env)
        self.policy = IndexPolicy(index=np.random.randint(100))
