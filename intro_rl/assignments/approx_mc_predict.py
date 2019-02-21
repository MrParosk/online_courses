# Approximate Monte Carlo for prediction

import numpy as np
from grid_world import standard_grid
from dp_policy_evaluation import print_value_func, print_policy
from monte_carlo_policy_evaluation_random import random_action, play_game

LEARNING_RATE = 0.001
GAMMA = 0.9
POSSIBLE_ACTIONS = ("U", "D", "L", "R")

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
        (1, 2): "U",
        (2, 1): "L",
        (2, 2): "U",
        (2, 3): "L",
    }

    # Initialize theta
    theta = np.random.randn(4) / 2

    # The model is V_hat = theta.dot(x), where x = [row, col, row*col, 1] (1 for bias term)
    def feature_extract(s):
        return np.array([s[0] - 1, s[1] - 1.5, s[0]*s[1] - 3, 1])

    t = 1.0
    for it in range(20000):
        if it % 100 == 0 and it != 0:
            t += 0.01

        alpha = LEARNING_RATE/t

        states_and_returns = play_game(grid, policy)

        for s, G in states_and_returns:
            x = feature_extract(s)
            V_hat = theta.dot(x)
            theta += alpha*(G - V_hat)*x

    V = {}
    states = grid.all_states()
    for s in states:
        if s in grid.actions:
            V[s] = theta.dot(feature_extract(s))
        else:
            V[s] = 0

    print("Value function")
    print_value_func(V, grid)
    print("\n")

    print("Policy")
    print_policy(policy, grid)
    print("\n")
