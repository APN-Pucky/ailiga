from pettingzoo.butterfly import knights_archers_zombies_v10
from pettingzoo.classic import rps_v2, tictactoe_v3
from pettingzoo.mpe import simple_spread_v2
from pqdm.processes import pqdm as ppqdm
from pqdm.threads import pqdm as tpqdm
from tianshou.env.pettingzoo_env import PettingZooEnv

from ailiga import all_fighters, env
from ailiga.APNPucky.DQNFighter.v0.DQNFighter_v0 import DQNFighter_v0
from ailiga.APNPucky.DQNFighter.v1.DQNFighter_v1 import DQNFighter_v1
from ailiga.APNPucky.DQNFighter.v2.DQNFighter_v2 import DQNFighter_v2
from ailiga.APNPucky.RandomFigher.v0.RandomFighter_v0 import RandomFighter_v0
from ailiga.tournament import Tournament


def _single_fight(t):
    t.fight()


class Season:
    def __init__(self, envs, agents, name=None, n_episodes=10000):
        self.name = name
        self.envs = envs
        self.agents = agents
        self.tournaments = []
        for e in self.envs:
            self.tournaments.append(
                Tournament(
                    e,
                    [a(e) for a in self.agents if a.valid_env(env.get_env_name(e))],
                    n_episodes,
                )
            )

    def fight(self):
        tpqdm(self.tournaments, _single_fight, n_jobs=1, desc=self.name)

    def as_rst(self):
        return "\n".join([t.as_rst() for t in self.tournaments])


default_season = Season(
    agents=all_fighters.get_all_fighters(),
    envs=env.get_all_envs(),
    name="default",
    n_episodes=10000,
)
