from pettingzoo.classic import rps_v2, tictactoe_v3
from tianshou.env.pettingzoo_env import PettingZooEnv

from ailiga.APNPucky.DQNFighter.DQNFighter_v0 import DQNFighter_v0
from ailiga.APNPucky.DQNFighter.DQNFighter_v1 import DQNFighter_v1
from ailiga.APNPucky.DQNFighter.DQNFighter_v2 import DQNFighter_v2
from ailiga.APNPucky.RandomFigher.RandomFighter_v0 import RandomFighter_v0
from ailiga.battle import Battle
from ailiga.env import simple_spread_v2


def test_random_random_battle():
    arena = Battle(simple_spread_v2, [DQNFighter_v0, RandomFighter_v0])
    print(arena.fight(1000))


test_random_random_battle()
