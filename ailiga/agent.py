class Agent:
    def __init__(self, lambda_env):
        self.lambda_env = lambda_env
        self.policy = None

    def get_policy(self):
        return self.policy

    def load_policy(self, path):
        pass

    def save_policy(self, path):
        pass

    def train(self):
        pass
