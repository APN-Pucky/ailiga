from tianshou.env.pettingzoo_env import PettingZooEnv

from ailiga.APNPucky.DQNFighter.DQNFighter_v0 import DQNFighter_v0
from ailiga.APNPucky.DQNFighter.DQNFighter_v1 import DQNFighter_v1
from ailiga.APNPucky.DQNFighter.DQNFighter_v2 import DQNFighter_v2
from ailiga.env import (
    knights_archers_zombies_v10,
    ledud_holdem_v4,
    rps_v2,
    simple_spread_v2,
    tictactoe_v3,
)

versions = [DQNFighter_v0, DQNFighter_v1, DQNFighter_v2]


def test_train_tictactoe_v3():
    for v in versions:
        f = v(tictactoe_v3)
        f.train()


def test_train_simple_spread_v2():
    for v in versions:
        f = v(simple_spread_v2)
        f.train()


def test_train_knights_archers_zombies_v10():
    for v in versions:
        f = v(knights_archers_zombies_v10)
        f.train()


def test_train_leduc_holdem_v4():
    for v in versions:
        f = v(ledud_holdem_v4)
        f.train()


def test_train_rps_v2():
    for v in versions:
        f = v(rps_v2)
        f.train()


# TODO does not work because observation_space is not a Box, but Discrete
# test_train_rps_v2()
test_train_leduc_holdem_v4()
test_train_knights_archers_zombies_v10()
test_train_tictactoe_v3()
test_train_simple_spread_v2()
