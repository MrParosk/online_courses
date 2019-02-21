# TD(0) for policy evaluation

import numpy as np
import matplotlib.pyplot as plt
from grid_world import standard_grid, negative_grid
from dp_policy_evaluation import print_value_func, print_policy

GAMMA = 0.9
ALPHA = 0.1
POSSIBLE_ACTIONS = ("U", "D", "L", "R")

def random_action(a, eps=0.1):
    p = np.random.random()
    if p < (1 - eps):
        return a
    else:
        return np.random.choice(POSSIBLE_ACTIONS)


def play_game(grid, policy):
    s = (2, 0)
    grid.set_state(s)
    states_and_rewards = [(s, 0)] # list of tuples of (state, reward)
    while not grid.game_over():
        a = policy[s]
        a = random_action(a)
        r = grid.move(a)
        s = grid.current_state()
        states_and_rewards.append((s, r))
    return states_and_rewards


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

    # Initialing V(s)
    V = {}
    states = grid.all_states()
    for s in states:
        V[s] = 0

    for it in range(1000):
        states_and_rewards = play_game(grid, policy)
        
        for t in range(len(states_and_rewards) - 1):
            s, _ = states_and_rewards[t]
            s2, r = states_and_rewards[t+1]
            V[s] = V[s] + ALPHA*(r + GAMMA*V[s2] - V[s])

    print("Value function")
    print_value_func(V, grid)
    print("\n")

    print("Policy")
    print_policy(policy, grid)
