import gym
from pettingzoo.classic import rps_v2, tictactoe_v3
from tianshou.env.pettingzoo_env import PettingZooEnv

from ailiga.APNPucky.DQNFighter.v0.DQNFighter_v0 import DQNFighter_v0
from ailiga.APNPucky.PPOFighter.PPOFighter_v0 import PPOFighter_v0
from ailiga.APNPucky.RandomFigher.v0.RandomFighter_v0 import RandomFighter_v0


def test_train_cartpole():
    f = RandomFighter_v0(lambda_env=lambda: gym.make("CartPole-v0"))
    f.train()


def test_train_tictactoe():
    f = RandomFighter_v0(lambda_env=lambda: PettingZooEnv(tictactoe_v3.env()))
    # f = PPOFighter_v0(lambda_env=lambda: gym.make("CartPole-v0"))
    f.train()


# def test_train_rps():
#    f = DQNFighter_v0(lambda_env=lambda: PettingZooEnv(rps_v2.env()))
#    f.train("dqn_agent_rps_v2.pth")


# test_train_rps()
