# Approximate TD(0) for prediction

import numpy as np
from grid_world import standard_grid
from dp_policy_evaluation import print_value_func, print_policy
from td_zero_prediction import play_game

GAMMA = 0.9
ALPHA = 0.1
POSSIBLE_ACTIONS = ("U", "D", "L", "R")

class Model:
    def __init__(self):
        self.theta = np.random.randn(4) / 2
    
    def feature_extract(self, s):
        return np.array([s[0] - 1, s[1] - 1.5, s[0]*s[1] - 3, 1])

    def predict(self, s):
        x = self.feature_extract(s)
        return self.theta.dot(x)

    def grad(self, s):
        return self.feature_extract(s)


if __name__ == "__main__":
    grid = standard_grid()

    print("Rewards")
    print_value_func(grid.rewards, grid)
    print("\n")

    policy = {
        (2, 0): "U",
        (1, 0): "U",
        (0, 0): "R",
        (0, 1): "R",
        (0, 2): "R",
        (1, 2): "R",
        (2, 1): "R",
        (2, 2): "R",
        (2, 3): "U",
    }

    model = Model()

    k = 1.0
    for it in range(20000):
        if it % 10 == 0 and it != 0:
            k += 0.01

        alpha = ALPHA/k

        states_and_rewards = play_game(grid, policy)

        for t in range(len(states_and_rewards) - 1):
            s, _ = states_and_rewards[t]
            s_prime, r = states_and_rewards[t+1]
            old_theta = model.theta.copy()
            
            if grid.is_terminal(s_prime):
                target = r
            else:
                target = r + GAMMA*model.predict(s_prime)
      
            model.theta += alpha*(target - model.predict(s))*model.grad(s)

    V = {}
    states = grid.all_states()
    for s in states:
        if s in grid.actions:
            V[s] = model.predict(s)
        else:
            V[s] = 0

    print("Value function")
    print_value_func(V, grid)
    print("\n")

    print("Policy")
    print_policy(policy, grid)
    print("\n")
