from pettingzoo.classic import rps_v2, tictactoe_v3
from tianshou.env.pettingzoo_env import PettingZooEnv

from ailiga.APNPucky.DQNAgent_v0 import DQNAgent
from ailiga.APNPucky.DQNFighter.DQNFighter_v0 import DQNFighter_v0
from ailiga.APNPucky.randomAgent_v0 import RandomAgent
from ailiga.APNPucky.RandomFighert.RandomFighter_v0 import RandomFighter_v0
from ailiga.arena import Arena


def test_random_arena():
    def lenv():
        return PettingZooEnv(rps_v2.env())

    arena = Arena(lenv, [RandomFighter_v0(lenv), RandomFighter_v0(lenv)])
    arena.fight(1)


def test_dqn_random_arena():
    def lenv():
        return PettingZooEnv(tictactoe_v3.env())

    arena = Arena(
        lenv,
        [
            RandomFighter_v0(lenv),
            DQNFighter_v0(lenv),
        ],
    )
    arena.fight(10000)
    arena = Arena(
        lenv,
        [
            RandomFighter_v0(lenv),
            RandomFighter_v0(lenv),
        ],
    )
    arena.fight(10000)


test_random_arena()
test_dqn_random_arena()
