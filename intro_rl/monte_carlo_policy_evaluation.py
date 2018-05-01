# Policy evaluation

import numpy as np
from grid_world import standard_grid
from dp_policy_evaluation import print_value_func, print_policy

GAMMA = 0.9
POSSIBLE_ACTIONS = ("U", "D", "L", "R")

def play_game(grid, policy):

    # Starting the game at random state since the policy is deterministic 
    start_states = list(grid.actions.keys())
    start_idx = np.random.choice(len(start_states))
    grid.set_state(start_states[start_idx])

    s = grid.current_state()

    # Saving the (states, reward)
    states_rewards = [(s, 0)]

    while not grid.game_over():
        a = policy[s]
        r = grid.move(a)
        s = grid.current_state()
        states_rewards.append((s, r))

    G = 0
    states_returns = []
    first = True

    # Calculate the returns (backwards from the terminal state)
    for s, r in reversed(states_rewards):

        # Skip first entry
        if first:
            first = False
        else:
            states_returns.append((s, G))
    
        G = r + GAMMA*G

    states_returns.reverse()
    return states_returns


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

    V = {}
    returns = {}

    for s in grid.all_states():
        if s in grid.actions:
            returns[s] = []
        else:
            V[s] = 0

    for _ in range(0, 100):
        states_returns = play_game(grid, policy)
        for s, G in states_returns:
            returns[s].append(G)
            V[s] = np.mean(returns[s])

    print("Value function")
    print_value_func(V, grid)
    print("\n")

    print("Policy")
    print_policy(policy, grid)
    print("\n")
