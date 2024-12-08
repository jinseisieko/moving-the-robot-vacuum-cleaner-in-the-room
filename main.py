import math
import random
import time
from pprint import pprint
from collections import deque
import numpy as np
import pygame
from functions import (
    update_robot_position,
    find_nearest_intersection,
    find_nearest_intersection_jit, check_warning_points,
)

# CONSTANTS
NUM_LIDAR_RAYS = 20
H = 0.05
LIDAR_ROTATION_TIME = 2.6527945914557116e-5  # one radian
LIDAR_TIME = 6.7e-8

# INPUT
print("All data is measured in meters (or radians)!!!")
room_width, room_height = tuple(map(float, input("Room size:").split()))
K = 400 / room_height
print("---Characteristics of the robot vacuum cleaner---")
robot_radius = float(input("The radius of robot:"))
wheel_diameter = float(input("The diameter of wheels:"))
wheel_distance = float(input("The distance between the wheels:"))
lidar_angle = float(input("The laser locator angle (radian):"))
delta_lidar_angle = lidar_angle / NUM_LIDAR_RAYS
robot_x, robot_y = tuple(map(float, input("The start position of robot:").split()))
robot_orientation = 0
cell_size = 3 * robot_radius
num_rows = int(room_width // cell_size + 1)
num_cols = int(room_height // cell_size + 1)

print("---Obstacles---")
num_obstacles = int(input("Number of obstacles:"))
obstacle_vertices = []
total_segments = 4

for i in range(num_obstacles):  # initial data entry
    print(f"Obstacle {i}")
    obstacle_vertices.append([])
    num_vertices = int(input("Number of vertices:"))
    total_segments += num_vertices
    for j in range(num_vertices):
        obstacle_vertices[-1].append(tuple(map(float, input("vertex x y:").split())))

# processed data for obstacles
obstacle_segments = np.zeros((total_segments, 2, 2), dtype=float)
obstacle_segments[0, :, :] = np.array(
    [
        [0, 0],
        [0, room_height],
    ]
)
obstacle_segments[1, :, :] = np.array(
    [
        [0, room_height],
        [room_width, room_height],
    ]
)
obstacle_segments[2, :, :] = np.array(
    [
        [room_width, room_height],
        [room_width, 0],
    ]
)
obstacle_segments[3, :, :] = np.array(
    [
        [room_width, 0],
        [0, 0],
    ]
)

segment_index = 0
for i, obstacle in enumerate(obstacle_vertices):
    for j, vertex in enumerate(obstacle):
        obstacle_segments[4 + segment_index, :, :] = np.array(
            [
                [vertex[0], vertex[1]],
                [
                    obstacle[(j + 1) % len(obstacle)][0],
                    obstacle[(j + 1) % len(obstacle)][1],
                ],
            ]
        )
        segment_index += 1

warning_points_data = np.zeros(
    (num_rows, num_cols, 1000, 3), dtype=float
)  # *BRAIN* OF ROBOT
available_ids = [[set(range(1000)) for _ in range(num_cols)] for _ in range(num_rows)]
pending_draw = deque()
lidar_full_time = (LIDAR_TIME + LIDAR_ROTATION_TIME) * NUM_LIDAR_RAYS
lidar_rays = []

def add_point_to_data(point):
    row = int(point[0] // cell_size)
    col = int(point[1] // cell_size)
    if len(available_ids[row][col]) == 0:
        print("WARN: No available IDs")
        return
    id_ = available_ids[row][col].pop()
    warning_points_data[row, col, id_, :2] = point
    warning_points_data[row, col, id_, 2] = 1
    pending_draw.append(point)


def simulation(delta_time):
    global robot_x, robot_y, robot_orientation
    if delta_time - lidar_full_time > 0:
        delta_time -= lidar_full_time
        for i in range(NUM_LIDAR_RAYS):
            alpha = lidar_angle/2 - delta_lidar_angle*i + robot_orientation
            point = find_nearest_intersection_jit(alpha, robot_x+robot_radius*math.cos(robot_orientation), robot_y+robot_radius*math.sin(robot_orientation), obstacle_segments)
            lidar_rays.append(np.array([[robot_x+robot_radius*math.cos(robot_orientation), robot_y+robot_radius*math.sin(robot_orientation)],point]))
            if not check_warning_points(point, warning_points_data, H, cell_size):
                add_point_to_data(point)
    robot_x, robot_y, robot_orientation = update_robot_position(
        robot_x,
        robot_y,
        robot_orientation,
        35,
        50,
        delta_time,
        wheel_diameter,
        wheel_distance / 2,
    )


pygame.init()
clock = pygame.time.Clock()
screen = pygame.display.set_mode((2 * K * room_width, 2 * K * room_height))
pygame.display.set_caption("Moving the Robot Vacuum Cleaner in the Room")
obstacle_surface = pygame.surface.Surface(
    (K * room_width, K * room_height), pygame.SRCALPHA
)
warning_points_surface = pygame.surface.Surface(
    (K * room_width, K * room_height), pygame.SRCALPHA
)

for i in range(total_segments):
    pygame.draw.line(
        obstacle_surface,
        "black",
        obstacle_segments[i, 0, :] * K,
        obstacle_segments[i, 1, :] * K,
    )

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
    lidar_rays = []
    simulation(1 / clock.get_fps() if clock.get_fps() != 0 else 0)

    while len(pending_draw) != 0:
        warning_point = pending_draw.pop()
        pygame.draw.circle(
            warning_points_surface,
            (255, 0, 0, 128),
            warning_point * K,
            robot_radius * K,
        )
    main_surface = pygame.surface.Surface((K * room_width, K * room_height))
    main_surface.fill("white")
    main_surface.blit(obstacle_surface, (0, 0))
    pygame.draw.circle(
        main_surface, "black", (K * robot_x, K * robot_y), robot_radius * K, 1
    )
    pygame.draw.line(
        main_surface,
        "black",
        (K * robot_x, K * robot_y),
        np.array([K * robot_x, K * robot_y])
        + np.array([math.cos(robot_orientation), math.sin(robot_orientation)])
        * K
        * robot_radius,
    )

    warning_surface = main_surface.copy()
    warning_surface.blit(warning_points_surface, (0, 0))
    for line in lidar_rays:
        pygame.draw.line(warning_surface, "red", line[0]*K, line[1]*K, 1)
    nearest_intersection = find_nearest_intersection_jit(
        robot_orientation, robot_x, robot_y, obstacle_segments
    )
    if nearest_intersection is not None:
        pygame.draw.circle(main_surface, "red", nearest_intersection * K, 5, 5)

    screen.blit(main_surface, (0, 0))
    screen.blit(warning_surface, (K * room_width, 0))
    pygame.display.update()
    print(clock.get_fps())
    clock.tick(120000)
