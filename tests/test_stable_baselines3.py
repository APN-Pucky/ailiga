import gym
from stable_baselines3 import A2C, PPO
from stable_baselines3.common.env_checker import check_env


def test_cartpole():
    env = gym.make("CartPole-v1")
    check_env(env)

    model = A2C("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=10_000)

    obs = env.reset()
    for i in range(1000):
        action, _state = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        env.render()
        if done:
            obs = env.reset()


def test_piston():
    import supersuit as ss
    from pettingzoo.butterfly import pistonball_v6
    from stable_baselines3 import PPO
    from stable_baselines3.ppo import CnnPolicy

    env = pistonball_v6.parallel_env(
        n_pistons=20,
        time_penalty=-0.1,
        continuous=True,
        random_drop=True,
        random_rotate=True,
        ball_mass=0.75,
        ball_friction=0.3,
        ball_elasticity=1.5,
        max_cycles=125,
    )
    env = ss.color_reduction_v0(env, mode="B")
    env = ss.resize_v1(env, x_size=84, y_size=84)
    env = ss.frame_stack_v1(env, 3)
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(env, 8, num_cpus=4, base_class="stable_baselines3")
    model = PPO(
        CnnPolicy,
        env,
        verbose=3,
        gamma=0.95,
        n_steps=256,
        ent_coef=0.0905168,
        learning_rate=0.00062211,
        vf_coef=0.042202,
        max_grad_norm=0.9,
        gae_lambda=0.99,
        n_epochs=5,
        clip_range=0.3,
        batch_size=256,
    )
    model.learn(total_timesteps=2000000)
    model.save("policy")

    # Rendering

    env = pistonball_v6.env()
    env = ss.color_reduction_v0(env, mode="B")
    env = ss.resize_v1(env, x_size=84, y_size=84)
    env = ss.frame_stack_v1(env, 3)

    model = PPO.load("policy")

    env.reset()
    for agent in env.agent_iter():
        obs, reward, done, info = env.last()
        act = model.predict(obs, deterministic=True)[0] if not done else None
        env.step(act)
        env.render()
import gym
import numpy as np
import supersuit as ss
from pettingzoo.mpe import simple_spread_v2
from pettingzoo.utils.conversions import (
    aec_to_parallel,
    turn_based_aec_to_parallel_wrapper,
)

def test_simple_spread():
    

    env = simple_spread_v2.parallel_env()
    # env = aec_to_parallel(env)
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(env, 1, num_cpus=1, base_class="stable_baselines3")

    # check_env(env)

    # Fix conversion gymnasium usage in petitngzoo but gym in PR fixed sb3
    env.action_space = gym.spaces.discrete.Discrete(5)
    env.observation_space = gym.spaces.box.Box(-np.inf, np.inf, (18,), np.float32)
    print(env.__dict__)

    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=10_000)

    obs = env.reset()
    for i in range(1000):
        action, _state = model.predict(obs, deterministic=True)
    obs, reward, termination, truncation, info = env.step(action)
    env.render()
    if done:
        obs = env.reset()

def test_sb():
    env = simple_spread_v2.parallel_env(N=2, local_ratio=0.5,max_cycles=10, continuous_actions=False)
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(env, 1, num_cpus=4,base_class='stable_baselines3')
    
    expert = PPO(
            policy="MlpPolicy",
            env=env,
            batch_size=64,
            ent_coef=0.0,
            learning_rate=0.0003,
            n_epochs=10,
            n_steps=64,
            tensorboard_log = './gail_single_gen',
        )
    expert.learn(100000)  # Note: set to 100000 to train a proficient expert
    #reward, r_std = evaluate_policy(expert, env, 50)
    #print(reward,r_std)

test_simple_spread()
