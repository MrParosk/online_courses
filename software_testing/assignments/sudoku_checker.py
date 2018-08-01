# SPECIFICATION:
#
# check_sudoku() determines whether its argument is a valid Sudoku
# grid. It can handle grids that are completely filled in, and also
# grids that hold some empty cells where the player has not yet
# written numbers.
#
# First, your code must do some sanity checking to make sure that its
# argument:
#
# - is a 9x9 list of lists
#
# - contains, in each of its 81 elements, an integer in the range 0..9
#
# If either of these properties does not hold, check_sudoku must
# return None.
#
# If the sanity checks pass, your code should return True if all of
# the following hold, and False otherwise:
#
# - each number in the range 1..9 occurs only once in each row
#
# - each number in the range 1..9 occurs only once in each column
#
# - each number the range 1..9 occurs only once in each of the nine
#   3x3 sub-grids, or "boxes", that make up the board
#
# This diagram (which depicts a valid Sudoku grid) illustrates how the
# grid is divided into sub-grids:
#
# 5 3 4 | 6 7 8 | 9 1 2
# 6 7 2 | 1 9 5 | 3 4 8
# 1 9 8 | 3 4 2 | 5 6 7
# ---------------------
# 8 5 9 | 7 6 1 | 4 2 3
# 4 2 6 | 8 5 3 | 7 9 1
# 7 1 3 | 9 2 4 | 8 5 6
# ---------------------
# 9 6 1 | 5 3 7 | 0 0 0
# 2 8 7 | 4 1 9 | 0 0 0
# 3 4 5 | 2 8 6 | 0 0 0
#
# Please keep in mind that a valid grid (i.e., one for which your
# function returns True) may contain 0 multiple times in a row,
# column, or sub-grid. Here we are using 0 to represent an element of
# the Sudoku grid that the player has not yet filled in.

# check_sudoku should return None
ill_formed = [[5, 3, 4, 6, 7, 8, 9, 1, 2],
              [6, 7, 2, 1, 9, 5, 3, 4, 8],
              [1, 9, 8, 3, 4, 2, 5, 6, 7],
              [8, 5, 9, 7, 6, 1, 4, 2, 3],
              [4, 2, 6, 8, 5, 3, 7, 9],  # <---
              [7, 1, 3, 9, 2, 4, 8, 5, 6],
              [9, 6, 1, 5, 3, 7, 2, 8, 4],
              [2, 8, 7, 4, 1, 9, 6, 3, 5],
              [3, 4, 5, 2, 8, 6, 1, 7, 9]]

# check_sudoku should return True
valid = [[5, 3, 4, 6, 7, 8, 9, 1, 2],
         [6, 7, 2, 1, 9, 5, 3, 4, 8],
         [1, 9, 8, 3, 4, 2, 5, 6, 7],
         [8, 5, 9, 7, 6, 1, 4, 2, 3],
         [4, 2, 6, 8, 5, 3, 7, 9, 1],
         [7, 1, 3, 9, 2, 4, 8, 5, 6],
         [9, 6, 1, 5, 3, 7, 2, 8, 4],
         [2, 8, 7, 4, 1, 9, 6, 3, 5],
         [3, 4, 5, 2, 8, 6, 1, 7, 9]]

# check_sudoku should return False
invalid = [[5, 3, 4, 6, 7, 8, 9, 1, 2],
           [6, 7, 2, 1, 9, 5, 3, 4, 8],
           [1, 9, 8, 3, 8, 2, 5, 6, 7],
           [8, 5, 9, 7, 6, 1, 4, 2, 3],
           [4, 2, 6, 8, 5, 3, 7, 9, 1],
           [7, 1, 3, 9, 2, 4, 8, 5, 6],
           [9, 6, 1, 5, 3, 7, 2, 8, 4],
           [2, 8, 7, 4, 1, 9, 6, 3, 5],
           [3, 4, 5, 2, 8, 6, 1, 7, 9]]

# check_sudoku should return True
easy = [[2, 9, 0, 0, 0, 0, 0, 7, 0],
        [3, 0, 6, 0, 0, 8, 4, 0, 0],
        [8, 0, 0, 0, 4, 0, 0, 0, 2],
        [0, 2, 0, 0, 3, 1, 0, 0, 7],
        [0, 0, 0, 0, 8, 0, 0, 0, 0],
        [1, 0, 0, 9, 5, 0, 0, 6, 0],
        [7, 0, 0, 0, 9, 0, 0, 0, 1],
        [0, 0, 1, 2, 0, 0, 3, 0, 6],
        [0, 3, 0, 0, 0, 0, 0, 5, 9]]

# check_sudoku should return True
hard = [[1, 0, 0, 0, 0, 7, 0, 9, 0],
        [0, 3, 0, 0, 2, 0, 0, 0, 8],
        [0, 0, 9, 6, 0, 0, 5, 0, 0],
        [0, 0, 5, 3, 0, 0, 9, 0, 0],
        [0, 1, 0, 0, 8, 0, 0, 0, 2],
        [6, 0, 0, 0, 0, 4, 0, 0, 0],
        [3, 0, 0, 0, 0, 0, 0, 1, 0],
        [0, 4, 0, 0, 0, 0, 0, 0, 7],
        [0, 0, 7, 0, 0, 0, 3, 0, 0]]


def is_9x9_grid(grid):
    if len(grid) != 9:
        return False

    for g in grid:
        if len(g) != 9:
            return False

    return True


def contains_only_0_to_9(grid):
    for g in grid:
        for element in g:
            if element not in range(0, 10):
                return False

    return True


def unique_elements_row(grid):
    for row in grid:
        unique_elements = []

        for element in row:
            if element not in unique_elements:
                unique_elements.append(element)
            else:
                # 0 is token for unfilled
                if element == 0:
                    continue
                else:
                    return False

    return True


def unique_elements_cols(grid):
    for col_index in range(0, 9):
        unique_elements = []
        for row_index in range(0, 9):

            element = grid[row_index][col_index]

            if element not in unique_elements:
                unique_elements.append(element)
            else:
                # 0 is token for unfilled
                if element == 0:
                    continue
                else:
                    return False

    return True


def unique_elements_subgrid(grid):
    for row_index in range(0, 3):
        for col_index in range(0, 3):
            
            # Grids of size 3x3
            start_row = row_index * 3
            end_row = (row_index + 1) * 3

            start_col = col_index * 3
            end_col = (col_index + 1) * 3

            unique_elements = []

            for row in range(start_row, end_row):
                for col in range(start_col, end_col):

                    element = grid[row][col]

                    if element not in unique_elements:
                        unique_elements.append(element)
                    else:
                        # 0 is token for unfilled
                        if element == 0:
                            continue
                        else:
                            return False
    return True


def check_sudoku(grid):
    # Your code here.
    if not is_9x9_grid(grid):
        return None

    if not contains_only_0_to_9(grid):
        return None

    if not unique_elements_row(grid):
        return False

    if not unique_elements_cols(grid):
        return False

    if not unique_elements_subgrid(grid):
        return False

    return True


if __name__ == "__main__":
    assert(check_sudoku(ill_formed) == None)  # --> None
    assert(check_sudoku(valid) == True)  # --> True
    assert(check_sudoku(invalid) == False)  # --> False
    assert(check_sudoku(easy) == True)  # --> True
    assert(check_sudoku(hard) == True)  # --> True
    print("All tests passed")
