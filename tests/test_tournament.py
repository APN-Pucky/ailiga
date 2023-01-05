from pettingzoo.classic import rps_v2, tictactoe_v3
from tianshou.env.pettingzoo_env import PettingZooEnv

from ailiga.APNPucky.DQNFighter.DQNFighter_v0 import DQNFighter_v0
from ailiga.APNPucky.DQNFighter.DQNFighter_v1 import DQNFighter_v1
from ailiga.APNPucky.DQNFighter.DQNFighter_v2 import DQNFighter_v2
from ailiga.APNPucky.RandomFigher.RandomFighter_v0 import RandomFighter_v0
from ailiga.tournament import Tournament


def lenv():
    return PettingZooEnv(tictactoe_v3.env())


def test_tournament():
    tournament = Tournament(
        lenv,
        [
            RandomFighter_v0(lenv),
            DQNFighter_v0(lenv),
            DQNFighter_v1(lenv),
            DQNFighter_v2(lenv),
        ],
        10,
    )
    tournament.fight()
    print(tournament.attacker_scores)
    print(tournament.defender_scores)


def test_tournament_tab():

    tournament = Tournament(
        lenv,
        [
            RandomFighter_v0(lenv),
            DQNFighter_v0(lenv),
            DQNFighter_v1(lenv),
            DQNFighter_v2(lenv),
        ],
        10,
    )
    tournament.fight()
    print(tournament.as_rst())
