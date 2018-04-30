# Policy iteration using exploring starts

import numpy as np
from grid_world import standard_grid, negative_grid
from dp_policy_evaluation import print_value_func, print_policy

GAMMA = 0.9
POSSIBLE_ACTIONS = ("U", "D", "L", "R")

def play_game(grid, policy):
    start_states = list(grid.actions.keys())
    start_idx = np.random.choice(len(start_states))
    grid.set_state(start_states[start_idx])

    s = grid.current_state()
    a = np.random.choice(POSSIBLE_ACTIONS)

    states_actions_rewards = [(s, a, 0)]
    seen_states = set()

    while True:
        old_s = grid.current_state()
        r = grid.move(a)
        s = grid.current_state()

        if s in seen_states:
            # Hack s.t. we don't end up in an infinitely long episode bumping into the wall
            states_actions_rewards.append((s, None, -100))
            break
        elif grid.game_over():
            states_actions_rewards.append((s, None, r))
            break
        else:
            a = policy[s]
            states_actions_rewards.append((s, a, r))
        seen_states.add(s)

    G = 0
    states_actions_returns = []
    first = True

    # Calculate the returns (backwards from the terminal state)
    for s, a, r in reversed(states_actions_rewards):
        if first:
            first = False
        else:
            states_actions_returns.append((s, a, G))
        G = r + GAMMA*G

    states_actions_returns.reverse()
    return states_actions_returns


def max_dict(d):
    # Returning argmax and max from a dictionary d
    max_key = None
    max_val = float("-inf")
    for k, v in d.items():
        if v > max_val:
            max_val = v
            max_key = k

    return max_key, max_val


if __name__ == "__main__":
    grid = negative_grid(step_cost=-0.1)

    print("Rewards")
    print_value_func(grid.rewards, grid)
    print("\n")

    # Initialize a random policy
    policy = {}
    for s in grid.actions.keys():
        policy[s] = np.random.choice(POSSIBLE_ACTIONS)

    Q = {}
    returns = {} 
    states = grid.all_states()

    # Initialize Q(s,a) and returns
    for s in states:
        if s in grid.actions: 
            Q[s] = {}
            for a in POSSIBLE_ACTIONS:
                Q[s][a] = 0
                returns[(s,a)] = []
        else:
            # terminal state
            pass

    for t in range(2000):
        # Generate an episode using pi
        states_actions_returns = play_game(grid, policy)

        for s, a, G in states_actions_returns:
            returns[(s, a)].append(G)
            Q[s][a] = np.mean(returns[(s, a)])

        # Updating policy
        for s in policy.keys():
            policy[s] = max_dict(Q[s])[0]

    print("Final policy")
    print_policy(policy, grid)
    print("\n")

    # Finding V
    V = {}
    for s, Qs in Q.items():
        V[s] = max_dict(Q[s])[1]

    print("Value function")
    print_value_func(V, grid)
