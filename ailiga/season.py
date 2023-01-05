from pettingzoo.butterfly import knights_archers_zombies_v10
from pettingzoo.classic import rps_v2, tictactoe_v3
from pettingzoo.mpe import simple_spread_v2
from pqdm.processes import pqdm as ppqdm
from pqdm.threads import pqdm as tpqdm
from tianshou.env.pettingzoo_env import PettingZooEnv

from ailiga import env
from ailiga.APNPucky.DQNFighter.DQNFighter_v0 import DQNFighter_v0
from ailiga.APNPucky.DQNFighter.DQNFighter_v1 import DQNFighter_v1
from ailiga.APNPucky.DQNFighter.DQNFighter_v2 import DQNFighter_v2
from ailiga.APNPucky.RandomFigher.RandomFighter_v0 import RandomFighter_v0
from ailiga.tournament import Tournament


def _single_fight(t):
    t.fight()


class Season:
    def __init__(self, envs, agents, name=None):
        self.name = name
        self.envs = envs
        self.agents = agents
        self.tournaments = []
        for e in self.envs:
            self.tournaments.append(
                Tournament(
                    e,
                    [a(e) for a in self.agents if a.valid_env(env.get_env_name(e))],
                    10000,
                )
            )

    def fight(self):
        tpqdm(self.tournaments, _single_fight, n_jobs=1, desc=self.name)

    def as_rst(self):
        return "\n".join([t.as_rst() for t in self.tournaments])


default_season = Season(
    agents=[
        RandomFighter_v0,
        DQNFighter_v0,
        DQNFighter_v1,
        DQNFighter_v2,
    ],
    envs=[
        lambda: PettingZooEnv(tictactoe_v3.env()),
        # lambda: PettingZooEnv(simple_spread_v2.env()),
        # lambda: PettingZooEnv(knights_archers_zombies_v10.env()),
    ],
    name="default",
)
