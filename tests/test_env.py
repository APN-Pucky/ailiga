from pettingzoo.butterfly import knights_archers_zombies_v10

env = knights_archers_zombies_v10.env()
env.reset()
for agent in env.agent_iter():
    observation, reward, termination, truncation, info = env.last()
    action = policy(observation, agent)
    env.step(action)
