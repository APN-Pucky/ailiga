import gym
from pettingzoo.classic import rps_v2, tictactoe_v3
from pettingzoo.mpe import simple_spread_v2
from tianshou.env.pettingzoo_env import PettingZooEnv

from ailiga.APNPucky.DQNFighter.v0.DQNFighter_v0 import DQNFighter_v0
from ailiga.APNPucky.PPOFighter.PPOFighter_v0 import PPOFighter_v0


def test_train_cartpole_v0():
    f = PPOFighter_v0(lambda_env=lambda: gym.make("CartPole-v0"))
    f.train()


def test_train_tictactoe_v3():
    f = PPOFighter_v0(lambda_env=lambda: PettingZooEnv(tictactoe_v3.env()))
    # f = PPOFighter_v0(lambda_env=lambda: gym.make("CartPole-v0"))
    f.train()


def test_train_simple_spread_v2():
    f = PPOFighter_v0(lambda_env=lambda: PettingZooEnv(simple_spread_v2.env()))
    f.train()


# test_train_cartpole_v0()

test_train_simple_spread_v2()

# not working yet?
# test_train_tictactoe_v3()
