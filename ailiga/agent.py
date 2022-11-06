class Agent:
    def __init__(self, env):
        self.env = env
        self.policy = None

    def get_policy(self):
        return self.policy

    def load_policy(self, path):
        pass

    def save_policy(self, path):
        pass

    def train(self):
        pass
