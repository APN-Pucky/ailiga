# from pettingzoo.classic import rps_v2, tictactoe_v3
from tianshou.env.pettingzoo_env import PettingZooEnv

from ailiga.APNPucky.DQNFighter.v0.DQNFighter_v0 import DQNFighter_v0
from ailiga.APNPucky.DQNFighter.v1.DQNFighter_v1 import DQNFighter_v1
from ailiga.APNPucky.DQNFighter.v2.DQNFighter_v2 import DQNFighter_v2
from ailiga.APNPucky.IndexFighter.v0.IndexFighter_v0 import IndexFighter_v0
from ailiga.APNPucky.RandomFigher.v0.RandomFighter_v0 import RandomFighter_v0
from ailiga.APNPucky.RandomIndexFighter.v0.RandomIndexFighter_v0 import (
    RandomIndexFighter_v0,
)
from ailiga.battle import Battle
from ailiga.env import rps_v2, tictactoe_v3


def test_rps_v2_random_random_battle():
    arena = Battle(rps_v2, [RandomFighter_v0, RandomFighter_v0])
    print(arena.fight(10))


def test_tictactoe_v3_random_random_battle():
    lenv = tictactoe_v3
    arena = Battle(lenv, [RandomFighter_v0(lenv), RandomFighter_v0(lenv)])
    print(arena.fight(10))


def test_tictactoe_v3_index_index_battle():
    lenv = tictactoe_v3
    arena = Battle(tictactoe_v3, [IndexFighter_v0, IndexFighter_v0])
    print(arena.fight(10))


def test_tictactoe_v3_random_index_index_battle():
    lenv = tictactoe_v3
    arena = Battle(lenv, [RandomIndexFighter_v0(lenv), IndexFighter_v0(lenv)])
    print(arena.fight(100))


def test_tictactoe_v3_random_index_random_battle():
    lenv = tictactoe_v3
    arena = Battle(lenv, [RandomIndexFighter_v0(lenv), RandomFighter_v0(lenv)])
    print(arena.fight(100))


def test_dqn_random_battle():
    lenv = tictactoe_v3

    arena = Battle(
        lenv,
        [
            DQNFighter_v0(lenv),
            DQNFighter_v0(lenv),
        ],
    )
    print(arena.fight(10))
    arena = Battle(
        lenv,
        [
            RandomFighter_v0(lenv),
            DQNFighter_v0(lenv),
        ],
    )
    print(arena.fight(10))
    arena = Battle(
        lenv,
        [
            RandomFighter_v0(lenv),
            RandomFighter_v0(lenv),
        ],
    )
    print(arena.fight(10))


test_tictactoe_v3_random_random_battle()
test_tictactoe_v3_index_index_battle()
test_tictactoe_v3_random_index_index_battle()
test_tictactoe_v3_random_index_random_battle()
