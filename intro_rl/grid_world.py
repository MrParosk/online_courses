import numpy as np

class Grid:
    def __init__(self, width, height, start_location):
        self.width = width
        self.height = height
        self.row = start_location[0]
        self.col = start_location[1]

    def set(self, rewards, actions):
        # Rewards should be a dict of: (row, col): reward
        # Actions should be a dict of: (row, col): list of possible actions
        self.rewards = rewards
        self.actions = actions

    def set_state(self, s):
        self.row = s[0]
        self.col = s[1]

    def current_state(self):
        return (self.row, self.col)

    def move(self, action):
        # check if legal move first
        if action in self.actions[(self.row, self.col)]:
            if action == 'U':
                self.row -= 1
            elif action == 'D':
                self.row += 1
            elif action == 'R':
                self.col += 1
            elif action == 'L':
                self.col -= 1
        return self.rewards.get((self.row, self.col), 0)

    def undo_move(self, action):
        # opposite to move
        if action == 'U':
          self.row += 1
        elif action == 'D':
          self.row -= 1
        elif action == 'R':
          self.col -= 1
        elif action == 'L':
          self.col += 1

    def is_terminal(self, s):
        return s not in self.actions

    def game_over(self):
        # Returns True if we are in a state where no actions are possible
        return (self.row, self.col) not in self.actions

    def all_states(self):
        return set(self.actions.keys()) | set(self.rewards.keys())


def standard_grid():
    """
        Define a grid that describes the reward for arriving at each state and possible actions at each state.
        Here, x means you can't go there, s means start position and the number implies reward at that state.
        The grid looks like this:
        .  .  .  1
        .  x  . -1
        s  .  .  .
    """

    g = Grid(3, 4, (2, 0))
    rewards = {(0, 3): 1, (1, 3): -1}
    actions = {
        (0, 0): ('D', 'R'),
        (0, 1): ('L', 'R'),
        (0, 2): ('L', 'D', 'R'),
        (1, 0): ('U', 'D'),
        (1, 2): ('U', 'D', 'R'),
        (2, 0): ('U', 'R'),
        (2, 1): ('L', 'R'),
        (2, 2): ('L', 'R', 'U'),
        (2, 3): ('L', 'U'),
    }
    g.set(rewards, actions)
    return g


def negative_grid(step_cost=-0.1):
    # Here we want to minimize the number of moves, i.e. penalize every move
    g = standard_grid()
    g.rewards.update({
        (0, 0): step_cost,
        (0, 1): step_cost,
        (0, 2): step_cost,
        (1, 0): step_cost,
        (1, 2): step_cost,
        (2, 0): step_cost,
        (2, 1): step_cost,
        (2, 2): step_cost,
        (2, 3): step_cost,
    })

    return g
