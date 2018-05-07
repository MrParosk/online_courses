# Temporal difference learning with Q-learning

import numpy as np
from grid_world import standard_grid, negative_grid
from dp_policy_evaluation import print_value_func, print_policy
from monte_carlo_policy_iteration_es import max_dict
from td_zero_prediction import random_action

GAMMA = 0.9
ALPHA = 0.1
POSSIBLE_ACTIONS = ("U", "D", "L", "R")

if __name__ == "__main__":
    grid = negative_grid(step_cost=-0.1)

    print("Rewards")
    print_value_func(grid.rewards, grid)
    print("\n")

    # Initialize Q(s,a)
    Q = {}
    states = grid.all_states()
    for s in states:
        Q[s] = {}
        for a in POSSIBLE_ACTIONS:
            Q[s][a] = 0

    # Keep track of how many times Q[s] has been updated
    update_counts = {}
    update_counts_sa = {}
    for s in states:
        update_counts_sa[s] = {}
        for a in POSSIBLE_ACTIONS:
            update_counts_sa[s][a] = 1.0

    t = 1.0
    for it in range(0, 10000):
        if it % 100 == 0 and it != 0:
            t += 1e-2

        s = (2, 0)
        grid.set_state(s)
        a, _ = max_dict(Q[s])

        while not grid.game_over():
            a = random_action(a, eps=0.5/t)
            r = grid.move(a)
            s_prime = grid.current_state()

            alpha = ALPHA / update_counts_sa[s][a]
            update_counts_sa[s][a] += 0.005

            old_qsa = Q[s][a]
            a2, max_q_s_a = max_dict(Q[s_prime])
            Q[s][a] = Q[s][a] + alpha*(r + GAMMA*max_q_s_a - Q[s][a])

            update_counts[s] = update_counts.get(s,0) + 1

            s = s_prime

    # Determine the policy from Q* and finding V*
    policy = {}
    V = {}
    for s in grid.actions.keys():
        a, max_q = max_dict(Q[s])
        policy[s] = a
        V[s] = max_q

    total = np.sum(list(update_counts.values()))
    for k, v in update_counts.items():
        update_counts[k] = v / total

    print("Update counts")
    print_value_func(update_counts, grid)
    print("\n")

    print("Value function")
    print_value_func(V, grid)
    print("\n")

    print("Policy")
    print_policy(policy, grid)
