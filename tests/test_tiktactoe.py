import tqdm
from pettingzoo.classic import tictactoe_v3

# from ailiga.APNPucky.PPOAgent import PPOAgent, train_ppo
from ailiga.APNPucky.randomAgent_v0 import RandomAgent
from ailiga.APNPucky.randomLegalAgent import RandomLegalAgent


def test_randoms():
    env = tictactoe_v3.env()
    agents = {
        "player_1": RandomAgent(env),
        "player_2": RandomLegalAgent(env),
    }
    scores = {
        "player_1": {"win": 0, "loss": 0, "draw": 0},
        "player_2": {"win": 0, "loss": 0, "draw": 0},
    }

    total_episodes = 10000

    for episode in tqdm.tqdm(range(total_episodes)):
        env.reset()
        for agent in env.agent_iter():
            observation, reward, termination, truncation, info = env.last()
            if termination:
                # accumulate scores
                rewards = env.rewards
                if rewards["player_1"] > rewards["player_2"]:
                    scores["player_2"]["loss"] += 1
                    scores["player_1"]["win"] += 1
                elif rewards["player_1"] < rewards["player_2"]:
                    scores["player_2"]["win"] += 1
                    scores["player_1"]["loss"] += 1
                else:
                    scores["player_1"]["draw"] += 1
                    scores["player_2"]["draw"] += 1
                break
            action = agents[agent].act(observation)
            env.step(action)
            # env.render()

    print("The results are:")
    print("Player 1: ", scores["player_1"])
    print("Player 2: ", scores["player_2"])


def test_train_ppo():
    env = tictactoe_v3.env()
    train_ppo(env)


test_randoms()
# test_train_ppo()
