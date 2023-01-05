import os
import pathlib
import time
import uuid

import numpy as np
import torch
from tianshou.env import DummyVectorEnv
from tianshou.utils import TensorboardLogger
from torch.utils.tensorboard import SummaryWriter

from ailiga import env
from ailiga.fighter import Fighter


class TrainedFighter(Fighter):
    """A trained fighter."""

    logdir = "log"
    traindir = "trained"
    agentindex = None

    training_num = 10
    test_num = 10

    train_envs = None
    test_envs = None

    def load(self, savefile=None):
        """Load the policy from a file."""
        if savefile is None:
            savefile = self.get_default_savefile()
        if os.path.isfile(savefile):
            self.policy.load_state_dict(torch.load(savefile))
            return True
        else:
            return False

    def get_default_savefile(self):
        """Get the default savefile name."""
        if self.user is None:
            raise ValueError("user must be set")
        name = (
            self.traindir
            + "/"
            + self.get_env_name()
            + "/"
            + self.get_user()
            + "/"
            + self.__class__.__name__
            + ".pth"
        )
        pathlib.Path(name).parent.mkdir(parents=True, exist_ok=True)
        return name

    def save(self, policy=None, savefile=None):
        """Save the policy."""
        if savefile is None:
            savefile = self.get_default_savefile()
        if policy is None:
            torch.save(
                policy.policies[self.env.agents[self.agentindex]].state_dict(), savefile
            )
        else:
            torch.save(self.policy.state_dict(), savefile)

    def reset(self):
        """Reset the policy and the environment."""
        for layer in self.policy.children():
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()

    def reward_metric(self, rews):
        """Pick the reward of the agent we are training."""
        return rews[:, self.agentindex]

    @classmethod
    def train_all(cls):
        for e in cls.compatible_envs():
            cls(env.get_env(e)).train()

    def train(self, seed=None, reset=True):
        self.train_envs = DummyVectorEnv(
            [self.lambda_env for _ in range(self.training_num)]
        )
        self.test_envs = DummyVectorEnv([self.lambda_env for _ in range(self.test_num)])

        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
            self.train_envs.seed(seed)
            self.test_envs.seed(seed)
            self.seed()

        if reset:
            self.reset()

        return None

    def get_logger(self):
        log_path = os.path.join(
            self.logdir,
            self.get_env_name(),
            self.get_user(),
            self.__class__.__name__,
            time.strftime("%Y-%m-%d_%H-%M-%S"),
        )
        writer = SummaryWriter(log_path)
        logger = TensorboardLogger(writer)
        return logger
