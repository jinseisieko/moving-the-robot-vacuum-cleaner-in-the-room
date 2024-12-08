from pprint import pprint

import numpy as np

# CONSTS

# INPUT
room_size_x, room_size_y = tuple(map(float, input("Room size:").split()))
print("---Characteristics of the robot vacuum cleaner---")
radius = float(input("Radius:"))
K = float(input("The distance between the wheels:")) / 2
loc_angle = float(input("The laser locator angle (radian):"))
x, y = tuple(map(float, input("The start position of robot:").split()))
size_cell = 3 * radius
rows = int(room_size_x // size_cell + 1)
cols = int(room_size_y // size_cell + 1)
print("---Obstacles---")
N_obst = int(input("Number of obstacles:"))
obst_ = []
N_segments = 0
for i in range(N_obst):  # initial data entry
    print(f"Obstacle {i}")
    obst_.append([])
    N_vert = int(input("Number of vertex:"))
    N_segments += N_vert
    for j in range(N_vert):
        obst_[-1].append(tuple(map(float, input("vertex x y:").split())))
obstacles = np.zeros((N_segments, 2, 2), dtype=float)  # processed data
for i, obs in enumerate(obst_):
    for j, point in enumerate(obs):
        obstacles[i + j, :, :] = np.array(
            [[point[0], point[1]],
             [obs[(j + 1) % len(obs)][0], obs[(j + 1) % len(obs)][1]]]
        )
pprint(obstacles)
# *BRAIN* OF ROBOT
data = np.zeros((rows, cols, 10000, 3), dtype=float)
