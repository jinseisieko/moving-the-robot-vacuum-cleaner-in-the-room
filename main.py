import math
import random
import time
from pprint import pprint
from collections import deque
import numpy as np
import pygame
from functions import (
    new_values,
    find_nearest_intersection,
    find_nearest_intersection_jit,
)

# CONSTS

# INPUT
print("All data is measured in meters (or radians)!!!")
room_size_x, room_size_y = tuple(map(float, input("Room size:").split()))
K = 400 / room_size_y
print("---Characteristics of the robot vacuum cleaner---")
radius = float(input("The radius of robot:"))
diameter = float(input("The diameter of wheels:"))
dist_wheels = float(input("The distance between the wheels:"))
loc_angle = float(input("The laser locator angle (radian):"))
x, y = tuple(map(float, input("The start position of robot:").split()))
alpha = 0
size_cell = 3 * radius
rows = int(room_size_x // size_cell + 1)
cols = int(room_size_y // size_cell + 1)
print("---Obstacles---")
N_obst = int(input("Number of obstacles:"))
obst_ = []
N_segments = 4
for i in range(N_obst):  # initial data entry
    print(f"Obstacle {i}")
    obst_.append([])
    N_vert = int(input("Number of vertex:"))
    N_segments += N_vert
    for j in range(N_vert):
        obst_[-1].append(tuple(map(float, input("vertex x y:").split())))
obstacles = np.zeros((N_segments, 2, 2), dtype=float)  # processed data
obstacles[0, :, :] = np.array(
    [
        [0, 0],
        [0, room_size_y],
    ]
)
obstacles[1, :, :] = np.array(
    [
        [0, room_size_y],
        [room_size_x, room_size_y],
    ]
)
obstacles[2, :, :] = np.array(
    [
        [room_size_x, room_size_y],
        [room_size_x, 0],
    ]
)
obstacles[3, :, :] = np.array(
    [
        [room_size_x, 0],
        [0, 0],
    ]
)
h = 0
for i, obs in enumerate(obst_):
    for j, point in enumerate(obs):
        obstacles[4 + h, :, :] = np.array(
            [
                [point[0], point[1]],
                [obs[(j + 1) % len(obs)][0], obs[(j + 1) % len(obs)][1]],
            ]
        )
        h += 1
data = np.zeros((rows, cols, 1000, 3), dtype=float)  # *BRAIN* OF ROBOT
ids = [[set(range(1000)) for _ in range(cols)] for _ in range(rows)]
wait_draw = deque()


def add_point_in_data(p):
    r = p[0] // size_cell
    c = p[1] // size_cell
    if len(ids[r][c]) == 0:
        print("WARN full")
        return
    id_ = ids[r][c].pop()
    data[r, c, id_, :3] = p
    data[r, c, id_, 3] = 1
    wait_draw.append(p)


def simulation(time):
    global x, y, alpha
    x, y, alpha = new_values(x, y, alpha, 10, 50, time, diameter, dist_wheels / 2)


pygame.init()
clock = pygame.time.Clock()
screen = pygame.display.set_mode((2 * K * room_size_x, 2 * K * room_size_y))
pygame.display.set_caption("moving the robot vacuum-cleaner in the room")
obst_surface = pygame.surface.Surface(
    (K * room_size_x, K * room_size_y), pygame.SRCALPHA
)
for i in range(N_segments):
    pygame.draw.line(
        obst_surface, "black", obstacles[i, 0, :] * K, obstacles[i, 1, :] * K
    )
while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()

    simulation(1 / clock.get_fps() if clock.get_fps() != 0 else 0)
    main_surface = pygame.surface.Surface((K * room_size_x, K * room_size_y))
    main_surface.fill("white")
    main_surface.blit(obst_surface, (0, 0))
    pygame.draw.circle(main_surface, "black", (K * x, K * y), radius * K, 1)
    pygame.draw.line(
        main_surface,
        "black",
        (K * x, K * y),
        np.array([K * x, K * y])
        + np.array([math.cos(alpha), math.sin(alpha)]) * K * radius,
    )

    pdg = find_nearest_intersection_jit(alpha, x, y, obstacles)
    if not pdg is None:
        pygame.draw.circle(main_surface, "red", pdg * K, 5, 5)
    screen.blit(main_surface, (0, 0))
    pygame.display.update()
    print(clock.get_fps())
    clock.tick(120000)
