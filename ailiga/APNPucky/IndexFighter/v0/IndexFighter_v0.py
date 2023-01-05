import itertools
import warnings
from typing import Any, Dict, Optional, Union

import gym
import numpy as np
import torch
from tianshou.data import Batch
from tianshou.policy import BasePolicy
from tianshou.utils import MultipleLRSchedulers

from ailiga.env import get_all_env_names
from ailiga.fighter import Fighter


class IndexPolicy(BasePolicy):
    """A fixed index agent used in multi-agent learning.

    It chooses an action at fixed index from the legal action.
    """

    def __init__(
        self,
        observation_space: Optional[gym.Space] = None,
        action_space: Optional[gym.Space] = None,
        action_scaling: bool = False,
        action_bound_method: str = "",
        lr_scheduler: Optional[
            Union[torch.optim.lr_scheduler.LambdaLR, MultipleLRSchedulers]
        ] = None,
        index=0,
    ) -> None:
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            action_scaling=action_scaling,
            action_bound_method=action_bound_method,
            lr_scheduler=lr_scheduler,
        )
        self.index = index

    def forward(
        self,
        batch: Batch,
        state: Optional[Union[dict, Batch, np.ndarray]] = None,
        **kwargs: Any,
    ) -> Batch:
        """Compute the fixed action over the given batch data.

        :return: A :class:`~tianshou.data.Batch` with "act" key, containing
            the fixed action.

        .. seealso::

            Please refer to :meth:`~tianshou.policy.BasePolicy.forward` for
            more detailed explanation.
        """
        # print(batch.terminated.__dict__,batch.done.__dict__)
        mask = batch.obs.mask
        # mask is a boolean array with shape (batch_size, action_dim)
        # mask[i,j] is True if action j is available for agent i
        # mask[i,j] is False if action j is unavailable for agent i

        # pick the first available action for each agent after the index
        act = np.full(mask.shape[0], 0)
        for i in range(mask.shape[0]):
            indexcount = 0
            # module the index by the number of available actions
            indexgoal = self.index % np.sum(mask[i, :])

            if not np.any(mask[i, :]):
                warnings.warn("No available actions for agent " + str(i))
            for j in itertools.cycle(range(mask.shape[1])):
                if mask[i, j]:
                    indexcount += 1
                    if indexcount > indexgoal:
                        act[i] = j
                        break

        return Batch(act=act)

    def learn(self, batch: Batch, **kwargs: Any) -> Dict[str, float]:
        """Since a random agent learns nothing, it returns an empty dict."""
        return {}


class IndexFighter_v0(Fighter):
    user = "APNPucky"

    @classmethod
    def compatible_envs(cls):
        return get_all_env_names()

    def __init__(self, env):
        super().__init__(env)
        self.policy = IndexPolicy(index=0)
