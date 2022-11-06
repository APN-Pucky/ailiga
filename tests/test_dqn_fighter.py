from pettingzoo.butterfly import knights_archers_zombies_v10
from pettingzoo.classic import leduc_holdem_v4, rps_v2, tictactoe_v3
from pettingzoo.mpe import simple_spread_v2
from tianshou.env.pettingzoo_env import PettingZooEnv

from ailiga.APNPucky.DQNFighter.DQNFighter_v0 import DQNFighter_v0
from ailiga.APNPucky.DQNFighter.DQNFighter_v1 import DQNFighter_v1
from ailiga.APNPucky.DQNFighter.DQNFighter_v2 import DQNFighter_v2


def test_train_tictactoe_v3():
    f = DQNFighter_v0(lambda_env=lambda: PettingZooEnv(tictactoe_v3.env()))
    f.train()
    f = DQNFighter_v1(lambda_env=lambda: PettingZooEnv(tictactoe_v3.env()))
    f.train()
    f = DQNFighter_v2(lambda_env=lambda: PettingZooEnv(tictactoe_v3.env()))
    f.train()


def test_train_simple_spread_v2():
    f = DQNFighter_v0(lambda_env=lambda: PettingZooEnv(simple_spread_v2.env()))
    f.train()


def test_train_knights_archers_zombies_v10():
    f = DQNFighter_v0(
        lambda_env=lambda: PettingZooEnv(knights_archers_zombies_v10.env())
    )
    f.train()


def test_train_leduc_holdem_v4():
    f = DQNFighter_v0(lambda_env=lambda: PettingZooEnv(leduc_holdem_v4.env()))
    f.train()


def test_train_rps_v2():
    f = DQNFighter_v0(lambda_env=lambda: PettingZooEnv(rps_v2.env()))
    f.train()


# TODO does not work because observation_space is not a Box, but Discrete
# test_train_rps_v2()
test_train_leduc_holdem_v4()
test_train_knights_archers_zombies_v10()
test_train_tictactoe_v3()
test_train_simple_spread_v2()
