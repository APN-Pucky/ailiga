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


class RandomIndexPolicy(IndexPolicy):
    def __init__(
        self,
        observation_space: Optional[gym.Space] = None,
        action_space: Optional[gym.Space] = None,
        action_scaling: bool = False,
        action_bound_method: str = "",
        lr_scheduler: Optional[
            Union[torch.optim.lr_scheduler.LambdaLR, MultipleLRSchedulers]
        ] = None,
        range=100,
    ) -> None:
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            action_scaling=action_scaling,
            action_bound_method=action_bound_method,
            lr_scheduler=lr_scheduler,
        )
        self.range = range
        self.index = np.random.randint(range)

    def forward(
        self,
        batch: Batch,
        state: Optional[Union[dict, Batch, np.ndarray]] = None,
        **kwargs: Any,
    ) -> Batch:
        # reroll index if all done
        if isinstance(batch.done, np.ndarray) and batch.done.all():
            self.index = np.random.randint(self.range)
        return super().forward(batch, state, **kwargs)


class RandomIndexFighter_v0(Fighter):
    user = "APNPucky"

    @classmethod
    def compatible_envs(cls):
        return get_all_env_names()

    def __init__(self, env):
        super().__init__(env)
        self.policy = RandomIndexPolicy(range=np.random.randint(100))
