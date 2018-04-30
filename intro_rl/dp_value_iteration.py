# Value iteration using dynamic programming

import numpy as np
from grid_world import negative_grid
from dp_policy_evaluation import print_value_func, print_policy

THRESHOLD = 1e-3
GAMMA = 0.9
POSSIBLE_ACTIONS = ("U", "D", "L", "R")

if __name__ == "__main__":
    grid = negative_grid()

    # Initializing policy randomly
    policy = {}
    for a in grid.actions.keys():
        policy[a] = np.random.choice(POSSIBLE_ACTIONS)

    print("Initial policy")
    print_policy(policy, grid)
    print("\n")

    # Initializing V(s)
    V = {}
    states = grid.all_states()
    for s in states:
        if s in grid.actions:
            # Random initialization
            V[s] = np.random.random()
        else:
            # Terminal state
            V[s] = 0

    while True:
        delta = 0
        for s in states:
            old_V = V[s]

            if s in policy:
                new_V = float("-inf")

                for a in POSSIBLE_ACTIONS:
                    grid.set_state(s)
                    r = grid.move(a)
                    v = r + GAMMA*V[grid.current_state()]

                    if v > new_V:
                        new_V = v

                V[s] = new_V
                delta = max(delta, np.abs(old_V - V[s]))

        if delta < THRESHOLD:
            break

    for s in policy.keys():

        best_a = None
        best_value = float("-inf")

        for a in POSSIBLE_ACTIONS:
            grid.set_state(s)
            r = grid.move(a)
            v = r + GAMMA * V[grid.current_state()]

            if v > best_value:
                best_value = v
                best_a = a

        policy[s] = best_a

    print("Final value function")
    print_value_func(V, grid)
    print("\n")

    print("Final policy")
    print_policy(policy, grid)
