from pettingzoo.classic import rps_v2
from tianshou.env.pettingzoo_env import PettingZooEnv

from ailiga.APNPucky.randomAgent_v0 import RandomAgent
from ailiga.arena import Arena


def test_random_arena():
    env = PettingZooEnv(rps_v2.env(max_cycles=1))
    arena = Arena(env, [RandomAgent(env), RandomAgent(env)])
    arena.fight(1)


test_random_arena()
