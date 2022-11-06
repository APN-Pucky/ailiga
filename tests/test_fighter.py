from pettingzoo.classic import rps_v2, tictactoe_v3
from tianshou.env.pettingzoo_env import PettingZooEnv

from ailiga.APNPucky.DQNAgent_v0 import DQNAgent
from ailiga.APNPucky.randomAgent_v0 import RandomAgent
from ailiga.arena import Arena


def test_train():
    f = DQNAgent(lambda_env=lambda: PettingZooEnv(tictactoe_v3.env()))
    f.train()


test_train()
