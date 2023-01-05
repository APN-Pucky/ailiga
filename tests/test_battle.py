from pettingzoo.classic import rps_v2, tictactoe_v3
from tianshou.env.pettingzoo_env import PettingZooEnv

from ailiga.APNPucky.DQNFighter.DQNFighter_v0 import DQNFighter_v0
from ailiga.APNPucky.DQNFighter.DQNFighter_v1 import DQNFighter_v1
from ailiga.APNPucky.DQNFighter.DQNFighter_v2 import DQNFighter_v2
from ailiga.APNPucky.RandomFigher.RandomFighter_v0 import RandomFighter_v0
from ailiga.battle import Battle


def test_random_random_battle():
    def lenv():
        return PettingZooEnv(rps_v2.env())

    arena = Battle(lenv, [RandomFighter_v0(lenv), RandomFighter_v0(lenv)])
    print(arena.fight(1000))


def test_dqn_random_battle():
    def lenv():
        return PettingZooEnv(tictactoe_v3.env())

    arena = Battle(
        lenv,
        [
            DQNFighter_v0(lenv),
            DQNFighter_v0(lenv),
        ],
    )
    print(arena.fight(1000))
    arena = Battle(
        lenv,
        [
            RandomFighter_v0(lenv),
            DQNFighter_v0(lenv),
        ],
    )
    print(arena.fight(1000))
    arena = Battle(
        lenv,
        [
            RandomFighter_v0(lenv),
            RandomFighter_v0(lenv),
        ],
    )
    print(arena.fight(1000))


test_random_random_battle()
test_dqn_random_battle()
