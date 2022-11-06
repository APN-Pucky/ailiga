from tianshou.data import Collector
from tianshou.env import DummyVectorEnv
from tianshou.policy import BasePolicy, DQNPolicy, MultiAgentPolicyManager, RandomPolicy


class Arena:
    def __init__(self, lambda_env, agents):
        self.env = lambda_env()
        self.agents = agents
        self.policies = [a.get_policy() for a in self.agents]
        self.env.reset()

    def fight(self, n_episodes=1, n_step=None, render=None):
        """Runs a number of episodes between two agents."""
        env = self.env
        policy = MultiAgentPolicyManager(self.policies, self.env)
        policy.eval()
        # policy.policies[agents[args.agent_id - 1]].set_eps(0.05)
        collector = Collector(
            policy, DummyVectorEnv([lambda: env]), exploration_noise=True
        )
        result = collector.collect(n_episode=n_episodes, n_step=n_step, render=render)
        rews, lens = result["rews"], result["lens"]
        # print(result)
        print(f"Final reward: {rews[:, 0].mean()}, length: {lens.mean()}")
        print(f"Final reward: {rews[:, 1].mean()}, length: {lens.mean()}")
