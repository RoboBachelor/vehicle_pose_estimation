import numpy as np


class item:
    def __init__(self):
        self.image_size = np.array([256, 192])
        self.heatmap_size = np.array([64, 48])

obj = item()

normal_distrib = [1, 0.7, 0.4, 0.2, 0.1, 0.05]

def generate_target_edge(obj, start, stop):
    target_edge = np.zeros([obj.heatmap_size[1], obj.heatmap_size[0]], dtype=float)

    feat_stride = obj.image_size / obj.heatmap_size

    start = np.array(start / feat_stride).astype(int)
    stop = np.array(stop / feat_stride).astype(int)

    # Using DDA algorithm to generate the line and push back to the queue
    dx = stop[0] - start[0]
    dy = stop[1] - start[1]
    k = dy / dx

    queue = []

    if abs(k) <= 1:
        if start[0] > stop[0]:
            tmp = start
            start = stop
            stop = tmp

        for cx in range(start[0], stop[0]):
            cy = round(start[1] + k * (cx - start[0]))
            queue.append((cx, cy, 0))
            target_edge[cy][cx] = 1
    else:
        if start[1] > stop[1]:
            tmp = start
            start = stop
            stop = tmp

        for cy in range(start[1], stop[1]):
            cx = round(start[0] + (1 / k) * (cy - start[1]))
            queue.append((cx, cy, 0))
            target_edge[cy][cx] = 1

    queue.append((stop[0], stop[1], 0))
    target_edge[stop[1]][stop[0]] = 1

    # u r d l
    dx = [0, 1, 0, -1]
    dy = [-1, 0, 1, 0]

    while len(queue) > 0:
        cx, cy, depth = queue.pop(0)
        for i in range(4):
            cx += dx[i]
            cy += dy[i]

            if cx >= 0 and cy >= 0 and cx < obj.heatmap_size[0] and cy < obj.heatmap_size[1]:
                if target_edge[cy][cx] == 0:
                    if depth < 5:
                        queue.append((cx, cy, depth + 1))
                        target_edge[cy][cx] = normal_distrib[depth + 1]

            cx -= dx[i]
            cy -= dy[i]

    return target_edge

import matplotlib.pyplot as plt

start = np.array([200, 10])
stop = np.array([200, 191])
plt.imshow(generate_target_edge(obj, start, stop))
plt.show()