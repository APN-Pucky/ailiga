import os
import pathlib
import time
import uuid

import numpy as np
import torch
from tianshou.utils import TensorboardLogger
from torch.utils.tensorboard import SummaryWriter

from ailiga.Fighter import Fighter


class TrainedFighter(Fighter):
    logdir = "log"
    traindir = "trained"
    user = None
    agent = None

    training_num = 10
    test_num = 10

    def load(self, savefile=None):
        if savefile is None:
            savefile = self.get_default_savefile()
        if os.path.isfile(savefile):
            self.policy.load_state_dict(torch.load(savefile))
            return True
        else:
            return False

    def get_user(self):
        return self.user

    def get_default_savefile(self):
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
        if savefile is None:
            savefile = self.get_default_savefile()
        if policy is None:
            torch.save(policy.policies[self.agent].state_dict(), savefile)
        else:
            torch.save(self.policy.state_dict(), savefile)

    def reset(self):
        for layer in self.policy.children():
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()

    def train(self, seed=None, reset=True):
        train_envs = DummyVectorEnv([self.lambda_env for _ in range(self.training_num)])
        test_envs = DummyVectorEnv([self.lambda_env for _ in range(self.test_num)])

        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
            train_envs.seed(seed)
            test_envs.seed(seed)
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
            time.strftime("%Y%m%d%H%M%S"),
        )
        writer = SummaryWriter(log_path)
        logger = TensorboardLogger(writer)
        return logger
