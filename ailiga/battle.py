import argparse
import multiprocessing as mp

import numpy as np
import tqdm
from tianshou.data import Collector
from tianshou.env import DummyVectorEnv, RayVectorEnv, SubprocVectorEnv
from tianshou.policy import BasePolicy, DQNPolicy, MultiAgentPolicyManager, RandomPolicy

from ailiga import env as menv
from ailiga.all_fighters import (
    get_all_fighters,
    get_fighter_by_name,
    get_fighters_from_list,
)


class Battle:
    """Runs a battle between two or more agents."""

    def __init__(self, lambda_env, agents):
        self.env = lambda_env()
        self.lambda_env = lambda_env
        if isinstance(agents[0], type):
            # agents are classes, not instances
            self.agents = [a(self.lambda_env) for a in agents]
        else:
            self.agents = agents
        self.policies = [a.get_policy() for a in self.agents]
        self.env.reset()
        self.rews = None
        self.lens = None
        if len(self.env.agents) != len(self.agents):
            raise ValueError(
                "Agents do not match environment: "
                + str(self.env.agents)
                + " vs "
                + str(self.agents)
            )

    def fight(self, n_episodes=1, n_step=None, render=None, n_jobs=None):
        """
        Runs a number of episodes between two agents.

        :param n_episodes: number of episodes to run
        :param n_step: number of steps per episode
        :param render: if True, render the environment
        :return: list of rewards
        """
        env = self.env
        policy = MultiAgentPolicyManager(self.policies, self.env)
        policy.eval()
        # policy.policies[agents[args.agent_id - 1]].set_eps(0.05)
        collector = Collector(
            policy,
            # DummyVectorEnv([lambda: env for _ in range(1)]),
            # SubprocVectorEnv([lambda: env for _ in range(10)]),
            SubprocVectorEnv(
                [
                    lambda: env
                    for _ in range(n_jobs if n_jobs is not None else mp.cpu_count())
                ]
            ),
            exploration_noise=True,
        )
        result = collector.collect(n_episode=n_episodes, n_step=n_step, render=render)
        self.rews, self.lens = result["rews"], result["lens"]
        return [self.rews[:, i].mean() for i in range(len(self.agents))]


def battle(
    a_fighter=None,
    a_env="tictactoe_v3",
    a_n_episodes=10000,
    a_n_steps=None,
    render=False,
    n_jobs=None,
    a_force=False,
):
    """Run a battle between agents."""
    e = a_env
    fghts = get_fighters_from_list(a_fighter)
    if not a_force:
        # get all fighters that are valid for the given env
        fghts = [a for a in fghts if a.valid_env(e)]
    b = Battle(menv.get_env(e), fghts)
    res = b.fight(
        n_episodes=a_n_episodes, n_step=a_n_steps, render=render, n_jobs=n_jobs
    )
    print("Env:", e)
    print("Fighters:", [a.get_name() for a in fghts])
    print("Rewards:", res)


def main():
    parser = argparse.ArgumentParser(description="Run a tournament between agents.")
    parser.add_argument("--n_episodes", type=int, default=10000)
    parser.add_argument("--n_step", type=int, default=None)
    parser.add_argument("--render", type=float, default=None)
    parser.add_argument("--n_jobs", type=int, default=None)

    parser.add_argument("--env", type=str, default="tictactoe_v3")
    parser.add_argument(
        "--fighter",
        type=str,
        nargs="+",
        default=[],
    )
    parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        help="force non checked constellations",
        default=False,
    )

    args = parser.parse_args()

    battle(
        args.fighter,
        args.env,
        args.n_episodes,
        args.n_step,
        args.render,
        args.n_jobs,
        args.force,
    )


if __name__ == "__main__":
    main()
