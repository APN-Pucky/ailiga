from pettingzoo.classic import rps_v2, tictactoe_v3
from tianshou.env.pettingzoo_env import PettingZooEnv

from ailiga.APNPucky.DQNFighter.DQNFighter_v0 import DQNFighter_v0
from ailiga.APNPucky.DQNFighter.DQNFighter_v1 import DQNFighter_v1
from ailiga.APNPucky.DQNFighter.DQNFighter_v2 import DQNFighter_v2


def test_train_tictactoe():
    f = DQNFighter_v0(lambda_env=lambda: PettingZooEnv(tictactoe_v3.env()))
    f.train()
    f = DQNFighter_v1(lambda_env=lambda: PettingZooEnv(tictactoe_v3.env()))
    f.train()
    f = DQNFighter_v2(lambda_env=lambda: PettingZooEnv(tictactoe_v3.env()))
    f.train()


test_train_tictactoe()
