import argparse
import multiprocessing as mp
import os

import numpy as np
from pqdm.processes import pqdm as ppqdm
from pqdm.threads import pqdm as tpqdm
from smpl_doc import doc
from smpl_io import io
from tianshou.data import Collector
from tianshou.env import DummyVectorEnv, SubprocVectorEnv
from tianshou.policy import BasePolicy, DQNPolicy, MultiAgentPolicyManager, RandomPolicy

from ailiga import env
from ailiga.all_fighters import get_all_fighters, get_fighter_by_name
from ailiga.APNPucky.RandomFigher.RandomFighter_v0 import RandomFighter_v0
from ailiga.battle import Battle
from ailiga.fighter import Fighter


def _single_fight(lambda_env, agents, n_episodes, n_step, i, j):
    battle = Battle(lambda_env, agents)
    rews = battle.fight(n_episodes, n_step)
    return i, j, rews


class Tournament:
    """A tournament is a series of battles between agents."""

    def __init__(
        self,
        lambda_env,
        fighters,
        n_episodes=1,
        n_step=None,
        fill=True,
    ):
        self.lambda_env = lambda_env
        # sort agents by name
        agents = sorted(fighters, key=lambda a: a.get_name())
        if isinstance(agents[0], type):
            # agents are classes, not instances
            self.agents = [a(self.lambda_env) for a in agents]
        else:
            self.agents = agents

        self.attacker_scores = np.zeros((len(self.agents), len(self.agents)))
        self.defender_scores = np.zeros((len(self.agents), len(self.agents)))
        self.n_episodes = n_episodes
        self.n_step = n_step
        self.fill = fill

    def get_name(self):
        return env.get_env_name(self.lambda_env) + " x" + str(self.n_episodes)

    def fight(self, n_jobs=1):
        """
        Battle between all fighters. Everyone fights everyone.

        :param n_episodes: number of episodes to run
        :param n_step: number of steps per episode
        :return: list of rewards
        """
        extra_agents = [RandomFighter_v0(self.lambda_env)] * (
            len(self.lambda_env().agents) - len(self.agents)
        )
        args = [
            {
                "lambda_env": self.lambda_env,
                "i": i,
                "j": j,
                "agents": [self.agents[i], self.agents[j], *extra_agents],
                "n_episodes": self.n_episodes,
                "n_step": self.n_step,
            }
            for i in range(len(self.agents))
            for j in range(len(self.agents))
        ]
        ret = tpqdm(
            args,
            _single_fight,
            n_jobs=n_jobs if n_jobs is not None else mp.cpu_count(),
            argument_type="kwargs",
            desc=self.get_name(),
        )
        for i, j, rews in ret:
            self.attacker_scores[i][j] = rews[0]
            self.defender_scores[i][j] = rews[1]
        return self.attacker_scores, self.defender_scores

    def as_rst(self, to_file=None):
        """Returns the scores as a restructured text table."""
        # print(self.attacker_scores)
        summed = [np.sum(a) for a in self.attacker_scores]
        ind_perm = sorted(range(len(summed)), key=lambda k: summed[k], reverse=True)
        # transpose and permute by ind_perm
        scores = [
            [
                self.attacker_scores[ind_perm[i]][ind_perm[j]]
                for i in range(len(self.agents))
            ]
            for j in range(len(self.agents))
        ]
        dic = {}
        dic[self.get_name()] = [
            str(i + 1)
            + ". @ "
            + "{0:.3g}".format(summed[ind_perm[i]] / len(self.agents))
            for i in range(len(self.agents))
        ]
        dic["defender\\\\attacker"] = [
            self.agents[ind_perm[i]].get_name() for i in range(len(self.agents))
        ]

        for i in sorted(
            range(len(self.agents)), key=lambda k: self.agents[ind_perm[k]].get_name()
        ):
            dic[self.agents[ind_perm[i]].get_name()] = scores[i]
        ret = doc.array_table(dic, tabs=0, init=True)
        if to_file is not None:
            io.write(
                to_file,
                ""
                # self.get_name()
                # + ", "
                + os.path.basename(to_file).split(".")[0]
                + "\n------------------------------\n\n"
                + ret,
            )
        return ret


def tournament(
    a_fighter=None,
    a_env="tictactoe_v3",
    to_file=None,
    a_n_episodes=10000,
    a_n_step=None,
):
    """Run a tournament between agents."""
    if len(a_fighter) == 0:
        # get all fighters that are valid for the given env
        fghts = [a for a in get_all_fighters() if a.valid_env(a_env)]

    else:
        fghts = [get_fighter_by_name(agent) for agent in a_fighter]

    t = Tournament(
        env.get_env(a_env),
        fghts,
        a_n_episodes,
        a_n_step,
    )
    t.fight()
    print(t.as_rst(to_file))


def main():
    parser = argparse.ArgumentParser(description="Run a tournament between agents.")
    parser.add_argument("--n_episodes", type=int, default=10000)
    parser.add_argument("--n_step", type=int, default=None)
    parser.add_argument("--env", type=str, default="tictactoe_v3")
    parser.add_argument(
        "--fighter",
        type=str,
        nargs="+",
        default=[],
    )
    parser.add_argument("--to_file", type=str, default=None, nargs="?")

    args = parser.parse_args()

    tournament(args.fighter, args.env, args.to_file, args.n_episodes, args.n_step)


if __name__ == "__main__":
    main()
