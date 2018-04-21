import numpy as np
from grid_world import standard_grid

THRESHOLD = 1e-3

def print_value_func(V, grid):
	for i in range(grid.width):
		print("---------------------------")
		for j in range(grid.height):
			v = V.get((i,j), 0)
			if v >= 0:
				print(" %.2f|" % v, end="")
			else:
				print("%.2f|" % v, end="")
		print("")


def print_policy(policy, grid):
	for i in range(grid.width):
		print("---------------------------")
		for j in range(grid.height):
			a = policy.get((i,j), ' ')
			print("  %s  |" % a, end="")
		print("")


if __name__ == "__main__":
	grid = standard_grid()
	states = grid.all_states()

	V = {}
	for s in states:
		V[s] = 0

	gamma = 1.0

	# Uniformly policy
	while True:
		delta = 0
		for s in states:
			old_V = V[s]
			if s in grid.actions:
				new_V = 0

				p_a = 1 / len(grid.actions[s])

				for a in grid.actions[s]:
					grid.set_state(s)
					r = grid.move(a)
					new_V += p_a*(r + gamma*V[grid.current_state()])

				V[s] = new_V
				delta = max(delta, np.abs(old_V - V[s]))

		if delta < THRESHOLD:
			break

	print("V(s) for uniformly policy")
	print_value_func(V, grid)
	print("\n")

	# Fixed policy 
	policy = {
		(2, 0): 'U',
		(1, 0): 'U',
		(0, 0): 'R',
		(0, 1): 'R',
		(0, 2): 'R',
		(1, 2): 'R',
		(2, 1): 'R',
		(2, 2): 'R',
		(2, 3): 'U',
  	}

	print("Fixed policy")
	print_policy(policy, grid)
	print("\n")

	V = {}
	for s in states:
		V[s] = 0

	gamma = 0.9

	while True:
		delta = 0
		for s in states:
			old_V = V[s]
			if s in policy:
				a = policy[s]
				new_V = 0
				grid.set_state(s)
				r = grid.move(a)
				new_V += r + gamma*V[grid.current_state()]

				V[s] = new_V
				delta = max(delta, np.abs(old_V - V[s]))

		if delta < THRESHOLD:
			break

	print("V(s) for fixed policy")
	print_value_func(V, grid)
