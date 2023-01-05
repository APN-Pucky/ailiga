import warnings

import gym
from tianshou.env.pettingzoo_env import PettingZooEnv

from ailiga import env


class Fighter:
    """Base class for all fighters."""

    user = None

    def __init__(self, lambda_env):
        self.lambda_env = lambda_env
        self.policy = None
        self.assert_env()

    @classmethod
    def get_user(cls):
        """Get the user name."""
        return cls.user

    @classmethod
    def compatible_envs(cls):
        """Return a list of compatible envs."""
        return []

    @classmethod
    def get_name(cls):
        """Get the name of the fighter."""
        return cls.get_user() + "/" + cls.__name__

    @classmethod
    def valid_env(cls, name):
        if cls.compatible_envs() is not True and name not in cls.compatible_envs():
            fighter_name = cls.get_name()
            return False
        return True

    def assert_env(self, name=None):
        """Assert that the env is compatible with the fighter."""
        if name is None:
            name = self.get_env_name()
        if not self.valid_env(name):
            fighter_name = self.get_name()
            warnings.warn(f"env {name} is not compatible with {fighter_name}")

    def get_env_name(self) -> str:
        """Get the name of the env."""
        return env.get_env_name(self.lambda_env)

    def get_policy(self):
        """Get the policy."""
        return self.policy
