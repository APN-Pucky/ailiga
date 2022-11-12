import warnings

import gym
from tianshou.env.pettingzoo_env import PettingZooEnv


class Fighter:
    def compatible_envs(self):
        return []

    def assert_env(self):
        name = self.get_env_name()
        if self.compatible_envs() is not True and name not in self.compatible_envs():
            warnings.warn(
                f"env {name} is not compatible with {self.__class__.__name__}"
            )

    def __init__(self, lambda_env):
        self.lambda_env = lambda_env
        self.policy = None
        self.assert_env()

    def get_env_name(self) -> str:
        env = self.lambda_env()
        if isinstance(env, gym.Env):
            return env.spec.id
        elif isinstance(env, PettingZooEnv):
            return env.env.metadata["name"]
        else:
            raise Exception("Unknown env type")

    def get_policy(self):
        return self.policy
