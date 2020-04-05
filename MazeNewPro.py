import random

import numpy as np
import matplotlib.pyplot as plt

from matplotlib import cm
from matplotlib.colors import ListedColormap

# Size
HEIGHT = 12
WIDTH = 12

# Param
GOAL = 1
GAMMA = 0.8
REWARD = -0.001
END_DELTA = 1e-5
MAX_ITER = 5000

# Enum
_BLOCK_TYPE = -2
_TRAP_TYPE = -1
_NORMAL_TYPE = 0
_GOAL_TYPE = 1

_LEFT = 0
_RIGHT = 1
_UP = 2
_DOWN = 3

# Color value
_COLOR_NORMAL_MIN = 0.25
_COLOR_NORMAL_MAX = 0.75
_COLOR_BLOCK = 0.8
_COLOR_TRAP = 0.85
_COLOR_ORIGIN = 0.9
_COLOR_GOAL = 0.95
_COLOR_PATH = 1.0

# Color map
viridis = cm.get_cmap('viridis', 200)
maze_colors = viridis(np.linspace(0, 1, 200))
maze_colors[156:165, :] = np.array([8 / 256, 8 / 256, 8 / 256, 1])
maze_colors[166:175, :] = np.array([79 / 256, 79 / 256, 72 / 256, 1])
maze_colors[176:185, :] = np.array([252 / 256, 250 / 256, 242 / 256, 1])
maze_colors[186:195, :] = np.array([251 / 256, 226 / 256, 81 / 256, 1])
maze_colors[196:, :] = np.array([208 / 256, 16 / 256, 76 / 256, 1])
maze_colormap = ListedColormap(maze_colors)

# Init map
maze_map = np.zeros((HEIGHT, WIDTH), dtype=int)
value_map = np.zeros((HEIGHT, WIDTH), dtype=float)
policy_map = np.zeros((HEIGHT, WIDTH), dtype=int)
mem_map = np.ones((HEIGHT, WIDTH), dtype=float)
view_map = np.zeros((HEIGHT, WIDTH), dtype=float)


def is_block(x, y):
    return maze_map[x, y] == _BLOCK_TYPE


def is_trap(x, y):
    return maze_map[x, y] == _TRAP_TYPE


def is_normal(x, y):
    return maze_map[x, y] == _NORMAL_TYPE


def is_goal(x, y):
    return maze_map[x, y] == _GOAL_TYPE


def get_value(x, y, x0, y0):
    if is_block(x, y):
        return value_map[x0, y0]
    elif is_trap(x, y):
        return value_map[1, 1]
    elif is_goal(x, y):
        return GOAL
    else:
        return value_map[x, y]


def cal_value(x, y):
    p = REWARD + GAMMA * np.array([
        0.6 * get_value(x, y - 1, x, y) + 0.2 * get_value(x - 1, y - 1, x, y) + 0.2 * get_value(x + 1, y - 1, x, y),
        0.6 * get_value(x, y + 1, x, y) + 0.2 * get_value(x - 1, y + 1, x, y) + 0.2 * get_value(x + 1, y + 1, x, y),
        0.6 * get_value(x - 1, y, x, y) + 0.2 * get_value(x - 1, y - 1, x, y) + 0.2 * get_value(x - 1, y + 1, x, y),
        0.6 * get_value(x + 1, y, x, y) + 0.2 * get_value(x + 1, y - 1, x, y) + 0.2 * get_value(x + 1, y + 1, x, y)
    ])
    value = np.max(p)
    policy = np.where(p == value)[0][0]
    return value, policy


def update_value():
    for i in range(HEIGHT):
        for j in range(WIDTH):
            if is_normal(i, j):
                value_map[i, j], policy_map[i, j] = cal_value(i, j)


def is_end():
    global mem_map
    if mem_map is None:
        return False

    delta = np.max(np.abs(value_map - mem_map))
    mem_map = value_map.copy()
    return delta < END_DELTA


def init_maze():
    # Init border
    maze_map[:1, :] = _BLOCK_TYPE
    maze_map[-1:, :] = _BLOCK_TYPE
    maze_map[:, :1] = _BLOCK_TYPE
    maze_map[:, -1:] = _BLOCK_TYPE

    # Init block
    maze_map[1, 2] = _BLOCK_TYPE
    maze_map[2, 6] = _BLOCK_TYPE
    maze_map[3, 6] = _BLOCK_TYPE
    maze_map[4, 1] = _BLOCK_TYPE
    maze_map[4, 2] = _BLOCK_TYPE
    maze_map[4, 3] = _BLOCK_TYPE
    maze_map[4, 4] = _BLOCK_TYPE
    maze_map[4, 5] = _BLOCK_TYPE
    maze_map[4, 6] = _BLOCK_TYPE
    maze_map[4, 7] = _BLOCK_TYPE
    maze_map[5, 3] = _BLOCK_TYPE
    maze_map[5, 5] = _BLOCK_TYPE
    maze_map[5, 7] = _BLOCK_TYPE
    maze_map[5, 8] = _BLOCK_TYPE
    maze_map[5, 9] = _BLOCK_TYPE
    maze_map[6, 3] = _BLOCK_TYPE
    maze_map[6, 5] = _BLOCK_TYPE
    maze_map[8, 6] = _BLOCK_TYPE
    maze_map[8, 7] = _BLOCK_TYPE
    maze_map[8, 8] = _BLOCK_TYPE
    maze_map[8, 9] = _BLOCK_TYPE
    maze_map[8, 10] = _BLOCK_TYPE
    maze_map[9, 2] = _BLOCK_TYPE
    maze_map[9, 3] = _BLOCK_TYPE
    maze_map[9, 4] = _BLOCK_TYPE
    maze_map[9, 5] = _BLOCK_TYPE
    maze_map[9, 6] = _BLOCK_TYPE
    maze_map[10, 8] = _BLOCK_TYPE

    # Init trap
    maze_map[2, 3] = _TRAP_TYPE

    # Init goal
    maze_map[-2, -2] = _GOAL_TYPE


def init_view():
    global view_map
    for i in range(HEIGHT):
        for j in range(WIDTH):
            if is_normal(i, j):
                view_map[i, j] = _COLOR_NORMAL_MIN
            elif is_block(i, j):
                view_map[i, j] = _COLOR_BLOCK
            elif is_trap(i, j):
                view_map[i, j] = _COLOR_TRAP

    view_map[1, 1] = _COLOR_ORIGIN
    view_map[-2, -2] = _COLOR_GOAL

    plt.ion()
    plt.figure(1, figsize=(10, 10))


def redraw():
    plt.clf()
    plt.imshow(view_map, cmap=maze_colormap, interpolation='none', vmin=0, vmax=_COLOR_PATH)
    for i in range(HEIGHT):
        for j in range(WIDTH):
            if is_normal(i, j):
                plt.text(j, i, '%.3f' % value_map[i, j], ha="center", va="center", color='w')
    plt.draw()
    plt.pause(0.5)


def get_normalize():
    value_list = []
    for i in range(HEIGHT):
        for j in range(WIDTH):
            if is_normal(i, j):
                value_list.append(value_map[i][j])
    return np.max(value_list) - np.min(value_list), np.min(value_list)


def update_view():
    global view_map
    k, b = get_normalize()
    update_map = np.where(maze_map == _NORMAL_TYPE, 0, 1)
    view_map = view_map * update_map + value_map
    redraw()


def rand_move():
    r = random.random()
    if r < 0.6:
        return 0
    elif r < 0.8:
        return -1
    else:
        return 1


def show_path():
    i = 1
    j = 1
    while True:
        view_map[i, j] = _COLOR_PATH
        redraw()

        if i < 0 or j < 0 or i >= HEIGHT or j >= WIDTH:
            raise Exception("Out of Bounds!")
        # elif is_block(i, j):
        #     raise Exception("Wrong Path!")
        elif is_goal(i, j):
            plt.ioff()
            plt.show()
            break
        # elif is_trap(i, j):
        #     i = 1
        #     j = 1
        elif is_normal(i, j):
            if policy_map[i, j] == 0:
                new_i = i + rand_move()
                new_j = j - 1
            elif policy_map[i, j] == 1:
                new_i = i + rand_move()
                new_j = j + 1
            elif policy_map[i, j] == 2:
                new_i = i - 1
                new_j = j + rand_move()
            else:
                new_i = i + 1
                new_j = j + rand_move()

            view_map[new_i, new_j] = _COLOR_ORIGIN
            redraw()

            if is_block(new_i, new_j):
                view_map[new_i, new_j] = _COLOR_BLOCK
            elif is_trap(new_i, new_j):
                view_map[new_i, new_j] = _COLOR_TRAP
                i = 1
                j = 1
            else:
                i = new_i
                j = new_j


init_maze()
init_view()
while not is_end():
    update_value()
    update_view()
print(value_map)
print(policy_map + maze_map)
show_path()
