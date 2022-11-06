from pettingzoo.classic import rps_v2, tictactoe_v3
from tianshou.env.pettingzoo_env import PettingZooEnv

from ailiga.APNPucky.DQNFighter.DQNFighter_v0 import DQNFighter_v0
from ailiga.Arena import Arena


def test_train_tictactoe():
    f = DQNFighter_v0(lambda_env=lambda: PettingZooEnv(tictactoe_v3.env()))
    f.train()


# def test_train_rps():
#    f = DQNFighter_v0(lambda_env=lambda: PettingZooEnv(rps_v2.env()))
#    f.train("dqn_agent_rps_v2.pth")


test_train_tictactoe()
# test_train_rps()
