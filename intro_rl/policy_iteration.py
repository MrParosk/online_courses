import numpy as np
from grid_world import standard_grid, negative_grid
from iterative_policy_evaluation import print_value_func, print_policy

THRESHOLD = 1e-3
GAMMA = 0.9
POSSIBLE_ACTIONS = ('U', 'D', 'L', 'R')

if __name__ == "__main__":
    grid = negative_grid()

    # Initializing policy randomly
    policy = {}
    for s in grid.actions.keys():
        policy[s] = np.random.choice(POSSIBLE_ACTIONS)

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
        # Policy evaluation
        while True:
            delta = 0
            for s in states:
                old_V = V[s]

                if s in policy:
                    a = policy[s]
                    grid.set_state(s)
                    r = grid.move(a)
                    V[s] = r + GAMMA*V[grid.current_state()]
                    delta = max(delta, np.abs(old_V - V[s]))

            if delta < THRESHOLD:
                break

        # Policy improvment
        policy_changed = True

        for s in states:
            if s in policy:
                old_a = policy[s]
                new_a = None

                best_value = float('-inf')

                # Go through all possible actions to find the best action
                for a in POSSIBLE_ACTIONS:
                    grid.set_state(s)
                    r = grid.move(a)
                    v = r + GAMMA*V[grid.current_state()]

                    if v > best_value:
                        best_value = v
                        new_a = a

                policy[s] = new_a
                if new_a != old_a:
                    policy_changed = False

        if policy_changed:
            break

    print("Final value function")
    print_value_func(V, grid)
    print("\n")

    print("Final policy")
    print_policy(policy, grid)
