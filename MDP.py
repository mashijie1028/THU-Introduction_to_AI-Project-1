from MazeView import *
# Basic Settings
# **********************************************************************************************************************
# **********************************************************************************************************************
# maze size and basic parameters
numRow = 10
numCol = 10
numStates = numRow * numCol
REWARD = -0.001
GAMMA = 0.8
epsilon = 1e-3
max_iteration = 500

# value parameters
V_ORIGIN = 0
V_BOUND = -1
V_TRAP = -1
V_BARRIER = -1
V_DESTINATION = 10

# directions
LEFT = 0
RIGHT = 1
UP = 2
DOWN = 3
LEFT_UP = 4
LEFT_DOWN = 5
RIGHT_UP = 6
RIGHT_DOWN = 7

# build the maze type
# origin      : 0
# destination : 1
# block       : -1
# trap        : -2
mazeType = np.zeros((numRow, numCol), dtype=int)
mazeType[0][0] = 0
mazeType[numRow - 1][numCol - 1] = 1


def maze_initialization(num_r, num_c):
    value = np.zeros((num_r, num_c))
    value[0][0] = V_ORIGIN
    value[num_r - 1][num_c - 1] = V_DESTINATION
    return value


def set_barrier(value, br, bc):
    mazeType[br][bc] = -1
    value[br][bc] = V_BARRIER


def is_barrier(br, bc):
    return mazeType[br][bc] == -1


def set_trap(value, tr, tc):
    mazeType[tr][tc] = -2
    value[tr][tc] = V_TRAP


def is_trap(tr, tc):
    return mazeType[tr][tc] == -2


def is_destination(dr, dc):
    if dr == numRow - 1 and dc == numCol - 1:
        return True
    else:
        return False


def is_bound(br, bc):
    return (br < 0) or (br > numRow - 1) or (bc < 0) or (bc > numCol - 1)


def get_row(row):
    if row < 0:
        return 0
    elif row > numRow - 1:
        return numRow - 1
    else:
        return row


def get_column(column):
    if column < 0:
        return 0
    elif column > numCol - 1:
        return numCol - 1
    else:
        return column


# set the policy for each state
def set_policy(policy, value, pr, pc, direction):
    policy[pr][pc][0] = value
    if direction is LEFT:
        policy[pr][pc][1] = LEFT
    elif direction is RIGHT:
        policy[pr][pc][1] = RIGHT
    elif direction is UP:
        policy[pr][pc][1] = UP
    elif direction is DOWN:
        policy[pr][pc][1] = DOWN


# get the value of each state
def get_value(value, row, column, direction):
    if is_bound(row, column):
        return value[get_row(row)][get_column(column)] + V_BOUND

    elif is_barrier(row, column):
        if direction == LEFT:
            return value[row][column + 1] + V_BARRIER
        elif direction == RIGHT:
            return value[row][column - 1] + V_BARRIER
        elif direction == UP:
            return value[row + 1][column] + V_BARRIER
        elif direction == DOWN:
            return value[row - 1][column] + V_BARRIER
        elif direction == LEFT_UP:
            return value[row + 1][column + 1] + V_BARRIER
        elif direction == LEFT_DOWN:
            return value[row - 1][column + 1] + V_BARRIER
        elif direction == RIGHT_UP:
            return value[row + 1][column - 1] + V_BARRIER
        elif direction == RIGHT_DOWN:
            return value[row - 1][column - 1] + V_BARRIER

    elif is_trap(row, column):
        return value[0][0] + V_TRAP

    else:
        return value[row][column]


# get max q_value of each state
def get_max_q_value(policy, q, qr, qc):
    # return max(q[qr][qc])
    max_q = -100000
    direction = -1
    max_dir = 0
    for i in q[qr][qc]:
        direction += 1
        if i > max_q:
            max_q = i
            max_dir = direction
    set_policy(policy, max_q, qr, qc, max_dir)
    return max_q


def is_iterate(v1, v2):
    return np.linalg.norm(v1 - v2, ord='fro') >= epsilon


# **********************************************************************************************************************
# **********************************************************************************************************************


# Instantiation
# **********************************************************************************************************************
# **********************************************************************************************************************
# Init MAZE
value1 = maze_initialization(numRow, numCol)  # value initialization
value_pre = value1  # value from the former iteration

# Set the barriers and traps for the maze
# set_barrier(value1, 0, 1)
# set_barrier(value1, 1, 5)
# set_barrier(value1, 2, 5)
# set_barrier(value1, 3, 0)
# set_barrier(value1, 3, 1)
# set_barrier(value1, 3, 2)
# set_barrier(value1, 3, 3)
# set_barrier(value1, 3, 4)
# set_barrier(value1, 3, 5)
# set_barrier(value1, 3, 6)
# set_barrier(value1, 4, 2)
# set_barrier(value1, 4, 4)
# set_barrier(value1, 4, 6)
# set_barrier(value1, 4, 7)
# set_barrier(value1, 4, 8)
# set_barrier(value1, 5, 2)
# set_barrier(value1, 5, 4)
# set_barrier(value1, 7, 5)
# set_barrier(value1, 7, 6)
# set_barrier(value1, 7, 7)
# set_barrier(value1, 7, 8)
# set_barrier(value1, 7, 9)
# set_barrier(value1, 8, 1)
# set_barrier(value1, 8, 2)
# set_barrier(value1, 8, 3)
# set_barrier(value1, 8, 4)
# set_barrier(value1, 8, 5)
# set_barrier(value1, 9, 7)
set_barrier(value1, 0, 1)
set_barrier(value1, 1, 1)
set_barrier(value1, 2, 1)
set_barrier(value1, 3, 1)
set_barrier(value1, 4, 1)
set_barrier(value1, 7, 6)
set_barrier(value1, 8, 6)
set_barrier(value1, 9, 6)
set_trap(value1, 2, 3)

# Init MazeView
maze_view = MazeView(mazeType)

q1 = np.zeros((numRow, numCol, 4))  # Q_value initialization
policy1 = np.zeros((numRow, numCol, 2))  # policy initialization
policy1[0][0][1] = RIGHT
q1[:, :, 0] = q1[:, :, 1] = q1[:, :, 2] = q1[:, :, 3] = value1

# directions for show the policy
directions = ['Left          ', 'Right         ', 'Up            ', 'Down          ', 'Back to origin',
              'Null          ']

# Iterate to update value, Q_value and policy
iteration = 0
while (iteration < 5) or ((iteration <= max_iteration) and is_iterate(value1, value_pre)):
    iteration += 1
    value_pre = value1
    for r in range(numRow):
        for c in range(numCol):
            # print(r, c)
            if not (r == numRow - 1 and c == numCol - 1) and not (is_barrier(r, c) or is_trap(r, c)):
                q1[r][c][LEFT] = GAMMA * ((0.6 * get_value(value1, r, c - 1, LEFT)) +
                                          (0.2 * get_value(value1, r - 1, c - 1, LEFT_UP)) +
                                          (0.2 * get_value(value1, r + 1, c - 1, LEFT_DOWN))) \
                                 + REWARD

                q1[r][c][RIGHT] = GAMMA * ((0.6 * get_value(value1, r, c + 1, RIGHT)) +
                                           (0.2 * get_value(value1, r - 1, c + 1, RIGHT_UP)) +
                                           (0.2 * get_value(value1, r + 1, c + 1, RIGHT_DOWN))) \
                                  + REWARD

                q1[r][c][UP] = GAMMA * ((0.6 * get_value(value1, r - 1, c, UP)) +
                                        (0.2 * get_value(value1, r - 1, c - 1, LEFT_UP)) +
                                        (0.2 * get_value(value1, r - 1, c + 1, RIGHT_UP))) \
                               + REWARD

                q1[r][c][DOWN] = GAMMA * ((0.6 * get_value(value1, r + 1, c, DOWN)) +
                                          (0.2 * get_value(value1, r + 1, c - 1, LEFT_DOWN)) +
                                          (0.2 * get_value(value1, r + 1, c + 1, RIGHT_DOWN))) \
                                 + REWARD

                value1[r][c] = get_max_q_value(policy1, q1, r, c)

    # Show the results of iterations
    if (iteration + 1) % 5 == 0:
        print("The value of iteration", iteration + 1, ":")
        for x in value1:
            print(x)
        print()

        print("The Q value of iteration", iteration + 1, ":")
        for y in q1:
            print(y)
        print()

        print("The Policy of iteration", iteration + 1, ":")
        for r in range(numRow):
            for c in range(numCol):
                if is_barrier(r, c) or is_destination(r, c):
                    print(directions[5], end='   ')
                elif is_trap(r, c):
                    print(directions[4], end='   ')
                else:
                    print(directions[int(policy1[r][c][1])], end='   ')
            print()

        # Update value and show
        maze_view.update_value(value1)

# Show path
maze_view.show_path(policy1[:, :, 1])
