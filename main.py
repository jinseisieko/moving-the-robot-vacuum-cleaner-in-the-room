import math
from pprint import pprint

import numpy as np
import pygame

# CONSTS

# INPUT
room_size_x, room_size_y = tuple(map(float, input("Room size:").split()))
K = 400 / room_size_y
print("---Characteristics of the robot vacuum cleaner---")
radius = float(input("Radius:"))
dist_wheels = float(input("The distance between the wheels:")) / 2
loc_angle = float(input("The laser locator angle (radian):"))
x, y = tuple(map(float, input("The start position of robot:").split()))
alpha = 0
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
            [
                [point[0], point[1]],
                [obs[(j + 1) % len(obs)][0], obs[(j + 1) % len(obs)][1]],
            ]
        )

data = np.zeros((rows, cols, 1000, 3), dtype=float)  # *BRAIN* OF ROBOT

def simulation(time):
    ...

pygame.init()
clock = pygame.time.Clock()
screen = pygame.display.set_mode((2 * K * room_size_x, 2 * K * room_size_y))
pygame.display.set_caption("moving the robot vacuum-cleaner in the room")

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()

    simulation(1 / clock.get_fps() if clock.get_fps() != 0 else 0)
    main_surface = pygame.surface.Surface((K * room_size_x, K * room_size_y))
    main_surface.fill("white")
    pygame.draw.circle(main_surface, "black", (K * x, K * y), radius * K, 1)
    pygame.draw.line(
        main_surface,
        "black",
        (K * x, K * y),
        np.array([K * x, K * y]) + np.array([math.cos(alpha), math.sin(alpha)]) * K * radius,
    )
    for i in range(N_segments):
        pygame.draw.line(main_surface, "black", obstacles[i, 0, :]*K, obstacles[i, 1, :]*K)

    screen.blit(main_surface, (0, 0))
    pygame.display.update()

    clock.tick(120)
