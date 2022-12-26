import heapq
from collections import deque
import numpy as np
import matplotlib.pyplot as plt

def plotLineLow(x0, y0, x1, y1):
    points = []
    dx = x1 - x0
    dy = y1 - y0
    yi = 1
    if dy < 0:
        yi = -1
        dy = -dy
    D = (2 * dy) - dx
    y = y0
    for x in range(x0, x1 + 1):
        points.append((y,x))
        if D > 0:
            y = y + yi
            D = D + (2 * (dy - dx))
        else:
            D = D + 2 * dy
    return points

def plotLineHigh(x0, y0, x1, y1):
    points = []
    dx = x1 - x0
    dy = y1 - y0
    xi = 1
    if dx < 0:
        xi = -1
        dx = -dx
    D = (2 * dx) - dy
    x = x0
    for y in range(y0, y1 + 1):
        points.append((y, x))
        if D > 0:
            x = x + xi
            D = D + (2 * (dx - dy))
        else:
            D = D + 2 * dx
    return points
def bline(y0, x0, y1, x1):
    points = []
    if abs(y1 - y0) < abs(x1 - x0):
        if x0 > x1:
            points.extend(plotLineLow(x1, y1, x0, y0))
        else:
            points.extend(plotLineLow(x0, y0, x1, y1))
    else:
        if y0 > y1:
            points.extend(plotLineHigh(x1, y1, x0, y0))
        else:
            points.extend(plotLineHigh(x0, y0, x1, y1))
    points = [[p[0] for p in points],[p[1] for p in points]]
    return points

class Node:
    def __init__(self, coords, parent=None):
        self.coords = coords
        self.parent = parent
        self.dist = np.inf
        self.f = np.inf

    def __lt__(self, other):
        return self.f < other.f

    def __gt__(self, obj):
        return ((self.f) > (obj.f))

    def __le__(self, obj):
        return ((self.f) <= (obj.f))

    def __ge__(self, obj):
        return ((self.f) >= (obj.f))

    def __eq__(self, other):
        is_equal = False
        if self.coords == other.coords:
            if self.parent is not None and other.parent is not None:
                is_equal = self.parent.coords == other.parent.coords
            else:
                if self.parent == other.parent:
                    is_equal = True

        return is_equal

    def __hash__(self):
        return hash(str(self.coords)+str(self.parent.coords if self.parent is not None else (-1,-1)))

    def __repr__(self):
        return str(self.coords) + " " + str(self.parent.coords)


class Lian:
    def __init__(self, distances, r=5, theta=45):
        self.grid = np.copy(distances).astype("float32")
        self.grid[self.grid == 1] = np.inf
        self.visited = set()
        self.visited_matrix = np.zeros_like(self.grid)
        self.queue = []

        self.lines_of_sight = {}
        self.circle = []

        self.r = r
        self.theta = theta
        self.start_angle = None


    def get_heuristic(self, start, finish):
        return ((start[0] - finish[0]) ** 2 + (start[1] - finish[1]) ** 2) ** 0.5

    def generate_lines_of_sight(self):
        y_n = self.circle[0]
        x_n = self.circle[1]

        for y,x in zip(y_n,x_n):
            self.lines_of_sight[(y,x)] = bline(0,0,y,x)


    def generate_circle(self):
        r = self.r
        # Form a minimal circle or radius r
        square = np.zeros((r * 2 + 1, r * 2 + 1))
        for x in range(-r, r + 1):
            for y in range(-r, r + 1):
                if (x ** 2 + y ** 2) ** 0.5 >= r and (x ** 2 + y ** 2) ** 0.5 <= r + 1:
                    square[r + x, r + y] = 1

        lower = np.where(np.diff(square, axis=0) == 1)
        upper = np.where(np.diff(square, axis=0) == -1)
        right = np.where(np.diff(square, axis=1) == 1)
        left = np.where(np.diff(square, axis=1) == -1)

        # get all the y and x coordinates of neighbours in a circle
        y_n = np.concatenate((lower[0] + 1, upper[0], left[0], right[0])) - r
        x_n = np.concatenate((lower[1], upper[1], left[1], right[1] + 1)) - r

        self.circle = [y_n, x_n]

    def get_neighbours(self, node):
        r = self.r
        theta = self.theta

        x_center = node.coords[1]
        y_center = node.coords[0]

        # get all the y and x coordinates of neighbours in a circle
        y_n = self.circle[0] + y_center
        x_n = self.circle[1] + x_center

        nodes = []

        # square2 = np.zeros((r * 2 + 1, r * 2 + 1))
        # for each pair of coordinates check if we can add them as neighbours
        for x, y in list(zip(x_n, y_n)):
            if -1 < x < self.grid.shape[1] and -1 < y < self.grid.shape[0]:
                if self.grid[y, x] >= 1:
                    # if we have a starting angle, than limit the scope by theta
                    line_of_sight = self.lines_of_sight[(y-y_center,x-x_center)].copy()
                    line_of_sight[0] += y_center
                    line_of_sight[1] += x_center

                    if np.sum(self.grid[line_of_sight[0], line_of_sight[1]] == 0) > 0:
                        # print(np.sum(self.grid[line_of_sight[0], line_of_sight[1]] == 0))
                        continue

                    if self.start_angle is not None:
                        if self.start_angle - theta <= np.arctan2(y - y_center, x - x_center) * 180 / np.pi <= self.start_angle + theta:
                            nodes.append(Node((y, x),node))
                            # square2[y-y_center+r,x-x_center+r] = 1
                    else:
                        nodes.append(Node((y, x),node))
                        # square2[y-y_center+r,x-x_center+r] = 1

        # plt.imshow(square2)
        # plt.show()
        nodes = set(nodes)
        nodes = [n for n in nodes if n not in self.visited]

        # print(len(nodes))
        # print(len(set(nodes)))
        # if len(nodes) != len(set(nodes)):
        #     print(nodes)

        return nodes

    def run(self, start, finish):
        # get initial circle
        self.generate_circle()
        self.generate_lines_of_sight()


        self.grid[start[0], start[1]] = 1
        start_node = Node(tuple(start))
        self.visited.add(start_node)


        neighbours = self.get_neighbours(start_node)

        for node in neighbours:
            self.grid[node.coords[0], node.coords[1]] = self.grid[start[0], start[1]] + self.r
            node.dist = self.get_heuristic(start_node.coords,node.coords)
            # node.dist = self.r
            node.f = node.dist + self.get_heuristic(node.coords, finish)
            heapq.heappush(self.queue, node)

        c = 0

        # continue while finish is not reached
        while len(self.queue) > 0:
            node = heapq.heappop(self.queue)

            if node in self.visited:
                continue

            self.visited.add(node)
            self.visited_matrix[node.coords[0], node.coords[1]] = 1

            self.start_angle = np.arctan2(node.coords[0] - node.parent.coords[0],
                                          node.coords[1] - node.parent.coords[1]) * 180 / np.pi

            if self.get_heuristic(node.coords, finish) < self.r:
                finish = Node(finish,node)
                break

            if node.coords[0] == finish[0] and node.coords[1] == finish[1]:
                finish = node
                break

            neighbours = self.get_neighbours(node)

            for n in neighbours:
                n.dist = node.dist + self.get_heuristic(node.coords, n.coords)
                n.f = n.dist + self.get_heuristic(n.coords, finish)
                heapq.heappush(self.queue, n)

            c += 1

            if c % 10000 == 0:
                print(len(self.visited))
                print(len(self.queue))
                # if c % 200000 == 0:
                #     plt.imshow(self.visited_matrix)
                #     plt.show()

        path = [finish.coords]

        prev = finish.parent
        while prev is not None:
            path.append(prev.coords)
            prev = prev.parent


        # plt.imshow(self.visited_matrix)
        # plt.show()

        return path
