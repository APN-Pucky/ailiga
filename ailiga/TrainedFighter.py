import os
import pathlib
import time
import uuid

import torch
from tianshou.utils import TensorboardLogger
from torch.utils.tensorboard import SummaryWriter

from ailiga.Fighter import Fighter


class TrainedFighter(Fighter):
    logdir = "log"
    traindir = "trained"
    user = None

    def load(self, savefile=None):
        if savefile is None:
            savefile = self.get_default_savefile()
        if os.path.isfile(savefile):
            map_location = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.policy.load_state_dict(torch.load(savefile, map_location=map_location))
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
