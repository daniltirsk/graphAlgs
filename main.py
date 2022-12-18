from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from lian import Lian
import cv2

def floodfill_notrec(matrix, x, y):
    dots_to_check = [[x,y]]
    while len(dots_to_check)>0:
        dot = dots_to_check.pop()
        x = dot[0]
        y = dot[1]

        if matrix[y, x] == -1:
            matrix[y, x] = 0
            if x > 0:
                dots_to_check.append([x-1,y])
            if x < matrix.shape[1] - 1:
                dots_to_check.append([x+1,y])
            if y > 0:
                dots_to_check.append([x,y-1])
            if y < matrix.shape[0] - 1:
                dots_to_check.append([x,y+1])


image = Image.open('imgs/karta-01.bmp')

startColor = [255,201,14]
finishColor = [237,28,36]
obstacleColor = [0,0,0]

city_map = np.array(image)

startInd = np.argwhere(np.sum(city_map == startColor,axis=2) == 3)
finishInd = np.argwhere(np.sum(city_map == finishColor,axis=2) == 3)

city_map = city_map.sum(axis=2)

city_map[city_map == 0] = 0
city_map[city_map != 0] = 1

start = startInd[0]
finish = finishInd[0]


lian_alg = Lian(city_map, r=5, theta=45)
shortest_path = lian_alg.run(start,finish)

city_map = np.array(image)

for i in range(len(shortest_path)-1):
    city_map = cv2.line(city_map, (shortest_path[i][1],shortest_path[i][0]), (shortest_path[i+1][1],shortest_path[i+1][0]), (255,0,0), 2)

plt.imshow(city_map)
plt.show()