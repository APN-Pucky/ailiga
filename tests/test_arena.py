from pettingzoo.classic import rps_v2, tictactoe_v3
from tianshou.env.pettingzoo_env import PettingZooEnv

from ailiga.APNPucky.DQNAgent_v0 import DQNAgent
from ailiga.APNPucky.randomAgent_v0 import RandomAgent
from ailiga.arena import Arena


def test_random_arena():
    def lenv():
        return PettingZooEnv(rps_v2.env())

    arena = Arena(lenv, [RandomAgent(lenv), RandomAgent(lenv)])
    arena.fight(1)


def test_dqn_random_arena():
    def lenv():
        return PettingZooEnv(tictactoe_v3.env())

    arena = Arena(
        lenv,
        [
            RandomAgent(lenv),
            DQNAgent(lenv, "dqn_agent.pth"),
        ],
    )
    arena.fight(10000)
    arena = Arena(
        lenv,
        [
            RandomAgent(lenv),
            RandomAgent(lenv),
        ],
    )
    arena.fight(10000)


test_random_arena()
test_dqn_random_arena()
