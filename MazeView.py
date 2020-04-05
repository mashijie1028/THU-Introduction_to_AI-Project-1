import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap
import numpy as np

_TYPE_NORMAL_MIN = 0.25
_TYPE_NORMAL_MAX = 0.75
_TYPE_WALL = 0.8
_TYPE_TRAP = 0.85
_TYPE_ORIGIN = 0.9
_TYPE_GOAL = 0.95
_TYPE_PATH = 1.0

viridis = cm.get_cmap('viridis', 200)
maze_colors = viridis(np.linspace(0, 1, 200))
maze_colors[156:165, :] = np.array([8/256, 8/256, 8/256, 1])
maze_colors[166:175, :] = np.array([79/256, 79/256, 72/256, 1])
maze_colors[176:185, :] = np.array([252/256, 250/256, 242/256, 1])
maze_colors[186:195, :] = np.array([251/256, 226/256, 81/256, 1])
maze_colors[196:, :] = np.array([208/256, 16/256, 76/256, 1])
maze_colormap = ListedColormap(maze_colors)

plt.ion()
plt.figure(1, figsize=(10, 10))


def get_max_and_min(value_map, block_map):
    value_list = []
    height, width = value_map.shape
    for i in range(height):
        for j in range(width):
            if block_map[i][j] == 1:
                value_list.append(value_map[i][j])
    return np.max(value_list), np.min(value_list)


class MazeView:
    def __init__(self, maze_type):
        self.height, self.width = maze_type.shape
        self.maze_map = np.empty((self.height + 2, self.width + 2), dtype=float)
        self.maze_map.fill(_TYPE_WALL)
        for i in range(self.height):
            for j in range(self.width):
                if maze_type[i][j] == -1:
                    self.maze_map[i + 1][j + 1] = _TYPE_WALL
                elif maze_type[i][j] == -2:
                    self.maze_map[i + 1][j + 1] = _TYPE_TRAP
                else:
                    self.maze_map[i + 1][j + 1] = _TYPE_NORMAL_MIN
        self.maze_map[1][1] = _TYPE_ORIGIN
        self.maze_map[self.height][self.width] = _TYPE_GOAL
        self.block_map = np.where(self.maze_map == _TYPE_NORMAL_MIN, 1, 0)
        self.value_map = np.empty((self.height, self.width))

    def update_value(self, maze_value):
        block_map_mini = self.block_map[1:-1, 1:-1]
        value_max, value_min = get_max_and_min(maze_value, block_map_mini)
        # print(value_max, value_min)
        self.maze_map = self.maze_map - self.block_map * self.maze_map
        # print(self.maze_map)
        self.maze_map[1:-1, 1:-1] = self.maze_map[1:-1, 1:-1] + block_map_mini * (
                (maze_value - value_min) / (value_max - value_min) * (_TYPE_NORMAL_MAX - _TYPE_NORMAL_MIN) + _TYPE_NORMAL_MIN)
        self.value_map = maze_value

        self.redraw()

    def show_path(self, policy):
        i = 0
        j = 0
        while True:
            # print(i, j, policy[i][j])
            self.maze_map[i+1][j+1] = _TYPE_PATH

            self.redraw()

            if i < 0 or j < 0 or i >= self.height or j >= self.width:
                raise Exception("Path out of bounds.")
            elif i == self.height - 1 and j == self.width - 1:
                plt.ioff()
                plt.show()
                break
            elif int(policy[i][j]) == 0:
                j = j - 1
            elif int(policy[i][j]) == 1:
                j = j + 1
            elif int(policy[i][j]) == 2:
                i = i - 1
            else:
                i = i + 1

    def redraw(self):
        plt.clf()
        plt.imshow(self.maze_map, cmap=maze_colormap, interpolation='none', vmin=0, vmax=_TYPE_PATH)
        for i in range(self.height):
            for j in range(self.width):
                if self.block_map[i+1][j+1] != 0:
                    plt.text(j+1, i+1, '%.2f' % self.value_map[i, j], ha="center", va="center", color='w')
        plt.draw()
        plt.pause(0.5)