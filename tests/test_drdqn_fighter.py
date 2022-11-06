from pettingzoo.classic import rps_v2, tictactoe_v3
from tianshou.env.pettingzoo_env import PettingZooEnv

from ailiga.APNPucky.DRDQNFighter.DRDQNFighter_v0 import DRDQNFighter_v0
from ailiga.Arena import Arena


def test_train_tictactoe():
    f = DRDQNFighter_v0(lambda_env=lambda: PettingZooEnv(tictactoe_v3.env()))
    f.train()


test_train_tictactoe()
