# Approximate SARSA for control

import numpy as np
from grid_world import negative_grid
from dp_policy_evaluation import print_value_func, print_policy
from monte_carlo_policy_iteration_es import max_dict
from td_zero_prediction import random_action

GAMMA = 0.9
ALPHA = 0.1
POSSIBLE_ACTIONS = ("U", "D", "L", "R")

class Model:
    def __init__(self):
        self.theta = np.random.randn(25) / np.sqrt(25)

    def feature_extract(self, s, a):
        # Encoding state and action to features
        return np.array([
            s[0] - 1              if a == "U" else 0,
            s[1] - 1.5            if a == "U" else 0,
            (s[0]*s[1] - 3)/3     if a == "U" else 0,
            (s[0]*s[0] - 2)/2     if a == "U" else 0,
            (s[1]*s[1] - 4.5)/4.5 if a == "U" else 0,
            1                     if a == "U" else 0,
            s[0] - 1              if a == "D" else 0,
            s[1] - 1.5            if a == "D" else 0,
            (s[0]*s[1] - 3)/3     if a == "D" else 0,
            (s[0]*s[0] - 2)/2     if a == "D" else 0,
            (s[1]*s[1] - 4.5)/4.5 if a == "D" else 0,
            1                     if a == "D" else 0,
            s[0] - 1              if a == "L" else 0,
            s[1] - 1.5            if a == "L" else 0,
            (s[0]*s[1] - 3)/3     if a == "L" else 0,
            (s[0]*s[0] - 2)/2     if a == "L" else 0,
            (s[1]*s[1] - 4.5)/4.5 if a == "L" else 0,
            1                     if a == "L" else 0,
            s[0] - 1              if a == "R" else 0,
            s[1] - 1.5            if a == "R" else 0,
            (s[0]*s[1] - 3)/3     if a == "R" else 0,
            (s[0]*s[0] - 2)/2     if a == "R" else 0,
            (s[1]*s[1] - 4.5)/4.5 if a == "R" else 0,
            1                     if a == "R" else 0,
            1
        ])

    def predict(self, s, a):
        x = self.feature_extract(s, a)
        return self.theta.dot(x)

    def grad(self, s, a):
        return self.feature_extract(s, a)

def getQs(model, s):
    # Chosing action based on a = argmax[a]{ Q(s,a) }
    Qs = {}
    for a in POSSIBLE_ACTIONS:
        Qs[a] = model.predict(s, a)
    return Qs


if __name__ == "__main__":
    grid = negative_grid(step_cost=-0.1)

    print("Rewards")
    print_value_func(grid.rewards, grid)
    print("\n")

    model = Model()

    t = 1.0
    for it in range(20000):
        if it % 100 == 0 and it != 0:
            t += 0.01

        alpha = ALPHA / t

        s = (2, 0)
        grid.set_state(s)
        Qs = getQs(model, s)

        a = max_dict(Qs)[0]
        a = random_action(a, eps=0.5/t)

        while not grid.game_over():
            r = grid.move(a)
            s_prime = grid.current_state()

            if grid.is_terminal(s_prime):
                model.theta += alpha*(r - model.predict(s, a))*model.grad(s, a)
            else:
                Qs_prime = getQs(model, s_prime)
                a_prime = max_dict(Qs_prime)[0]
                a_prime = random_action(a_prime, eps=0.5/t)

                model.theta += alpha*(r + GAMMA*model.predict(s_prime, a_prime) - model.predict(s, a))*model.grad(s, a)
        
                s = s_prime
                a = a_prime

    # Finding the policy and V* from Q*
    policy = {}
    V = {}
    Q = {}
    for s in grid.actions.keys():
        Qs = getQs(model, s)
        Q[s] = Qs
        a, max_q = max_dict(Qs)
        policy[s] = a
        V[s] = max_q

    print("Value function")
    print_value_func(V, grid)
    print("\n")

    print("Policy")
    print_policy(policy, grid)
    print("\n")
