import heapq
from collections import deque
import numpy as np
import matplotlib.pyplot as plt

from dataclasses import dataclass, field
from typing import Any


class Node:
    def __init__(self, coords, parent=None):
        self.coords = coords
        self.parent = parent
        self.dist = np.inf

    def __lt__(self, other):
        return self.dist < other.dist


class Lian:
    def __init__(self, distances, r=5, theta=45):
        self.grid = np.copy(distances).astype("float32")
        self.grid[self.grid == 1] = np.inf
        self.visited = np.zeros_like(distances)
        self.queue = []

        self.r = r
        self.theta = theta
        self.start_angle = None


    def get_heuristic(self, start, finish):
        return ((start[0] - finish[0]) ** 2 + (start[1] - finish[1]) ** 2) ** 0.5

    def get_neighbours(self, node):
        r = self.r
        theta = self.theta

        x_center = node.coords[1]
        y_center = node.coords[0]

        square = np.zeros((r * 2 + 1, r * 2 + 1))
        for x in range(-r, r + 1):
            for y in range(-r, r + 1):
                if (x ** 2 + y ** 2) ** 0.5 >= r and (x ** 2 + y ** 2) ** 0.5 >= r:
                    square[r + x, r + y] = 1

        lower = np.where(np.diff(square, axis=0) == 1)
        upper = np.where(np.diff(square, axis=0) == -1)
        right = np.where(np.diff(square, axis=1) == 1)
        left = np.where(np.diff(square, axis=1) == -1)

        y_n = np.concatenate((lower[0] + 1, upper[0], left[0], right[0])) - r + y_center
        x_n = np.concatenate((lower[1], upper[1], left[1], right[1] + 1)) - r + x_center

        nodes = []

        for x, y in list(zip(x_n, y_n)):
            if -1 < x < self.grid.shape[1] and -1 < y < self.grid.shape[0]:
                if self.visited[y, x] == 0 and self.grid[y, x] >= 1:
                    if self.start_angle is not None:
                        # print(np.arctan2(y - y_center, x - x_center) * 180 / np.pi,self.start_angle+theta, (y - y_center, x - x_center))
                        if self.start_angle - theta <= np.arctan2(y - y_center, x - x_center) * 180 / np.pi <= self.start_angle + theta:
                            nodes.append((y, x))
                    else:
                        nodes.append((y, x))

        return nodes

    def get_prev_node(self, node):
        pass

    def run(self, start, finish):
        self.grid[start[0], start[1]] = 1
        self.visited[start[0], start[1]] = 1
        start_node = Node(start)

        neighbours = self.get_neighbours(start_node)

        for n in neighbours:
            self.grid[n[0], n[1]] = self.grid[start[0], start[1]] + self.r
            self.visited[n[0], n[1]] = 1
            node = Node(n, start_node)
            node.dist = self.get_heuristic(n, finish)
            heapq.heappush(self.queue, node)

        c = 0

        node = heapq.heappop(self.queue)
        self.start_angle = np.arctan2(node.coords[0] - start_node.coords[0], node.coords[1] - start_node.coords[1]) * 180 / np.pi
        # print(self.start_angle)
        self.visited[node.coords[0],node.coords[1]] = 2


        while len(self.queue) > 0:
            if self.get_heuristic(node.coords, finish) < self.r:
                finish = node
                break

            if node.coords[0] == finish[0] and node.coords[1] == finish[1]:
                finish = node
                break

            neighbours = self.get_neighbours(node)

            for n in neighbours:
                if self.grid[n[0], n[1]] > self.grid[node.coords[0], node.coords[1]] + self.r:
                    self.grid[n[0], n[1]] = self.grid[node.coords[0], node.coords[1]] + self.r
                    self.visited[n[0], n[1]] = 1
                    n_node = Node(n, node)
                    n_node.dist = self.get_heuristic(n, finish) + self.grid[n[0], n[1]]
                    heapq.heappush(self.queue, n_node)

            prev_node = node
            node = heapq.heappop(self.queue)
            self.start_angle = np.arctan2(node.coords[0] - prev_node.coords[0],
                                          node.coords[1] - prev_node.coords[1]) * 180 / np.pi

            c += 1

            # if c == 1000:
            #     plt.imshow(self.visited)
            #     plt.show()
            #     break

        path = [finish.coords]
        prev = finish.parent
        while prev is not None:
            path.append(prev.coords)
            prev = prev.parent


        # plt.imshow(self.visited)
        # plt.show()

        return path
