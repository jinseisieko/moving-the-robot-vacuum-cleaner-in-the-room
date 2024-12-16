import math
import random
from collections import deque
import numpy as np
import pygame
from functions import (
    update_robot_position_jit,
    find_nearest_intersection_jit,
    check_area_points_jit,
    count_intersections_jit,
    check_cleared_point_jit,
)

H = 0.15
LIDAR_ROTATION_TIME = 2.6527945914557116e-5  # one radian
LIDAR_TIME = 6.7e-8
BLUE_RADIUS_FACTOR = 1.5
RED_RADIUS_FACTOR = 1.2
NUM_BLUE_WAIT = 300
TIMER_TRAJECTORY = 1
YELLOW = "#a09a07"

print("All data is measured in meters (or radians)!!!")
room_width, room_height = tuple(map(float, input("Room size:").split()))
K = 400 / room_height
print("---Characteristics of the robot vacuum cleaner---")
robot_radius = float(input("The radius of robot:"))
wheel_diameter = float(input("The diameter of wheels:"))
wheel_distance = float(input("The distance between the wheels:"))
lidar_angle = float(input("The laser locator angle (radian):"))
delta_lidar_angle = 1 / 30
robot_x, robot_y = tuple(map(float, input("The start position of robot:").split()))
robot_orientation = 0
cell_size = 2 * robot_radius
num_rows = int(room_width // cell_size + 1)
num_cols = int(room_height // cell_size + 1)

print("---Obstacles---")
num_obstacles = int(input("Number of obstacles:"))
obstacle_vertices = []
total_segments = 4

num_lidar_rays = int((1 / delta_lidar_angle) * lidar_angle)

for i in range(num_obstacles):  # initial data entry
    print(f"Obstacle {i}")
    obstacle_vertices.append([])
    num_vertices = int(input("Number of vertices:"))
    total_segments += num_vertices
    for j in range(num_vertices):
        obstacle_vertices[-1].append(tuple(map(float, input("vertex x y:").split())))

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
yellow_points = np.zeros((num_rows, num_cols, 3), dtype=np.float64)
for i in range(num_rows):
    for j in range(num_cols):
        yellow_points[i][j] = np.array(
            [cell_size / 2 + i * cell_size, cell_size / 2 + j * cell_size, 1],
            dtype=np.float64,
        )
for i in range(num_rows):
    for j in range(num_cols):
        if (
            count_intersections_jit(
                robot_x,
                robot_y,
                yellow_points[i][j][0],
                yellow_points[i][j][1],
                obstacle_segments,
            )
            % 2
            != 0
        ):
            yellow_points[i][j][2] -= 2.0

red_points_data = np.zeros((num_rows + 1, num_cols + 1, 1000, 3), dtype=float)
blue_points_data = np.zeros((num_rows + 1, num_cols + 1, 1000, 3), dtype=float)
green_points_data = np.zeros((num_rows + 1, num_cols + 1, 1000, 3), dtype=float)
available_ids_red_points = [
    [set(range(1000)) for _ in range(num_cols)] for _ in range(num_rows)
]
available_ids_blue_points = [
    [set(range(1000)) for _ in range(num_cols)] for _ in range(num_rows)
]
available_ids_green_points = [
    [set(range(1000)) for _ in range(num_cols)] for _ in range(num_rows)
]
blue_queue = deque()
pending_draw_yellow_points = deque()
pending_draw_red_points = deque()
pending_draw_blue_points = deque()
pending_draw_green_points = deque()
lidar_full_time = (LIDAR_TIME + LIDAR_ROTATION_TIME) * num_lidar_rays
moving_full_time = 5e-3
lidar_rays = []
T = 0
timer_trajectory = TIMER_TRAJECTORY
last_point = np.array([robot_x, robot_y])


def add_point_to_yellow_points(point):
    row = int(point[0] // cell_size)
    col = int(point[1] // cell_size)
    if yellow_points[row, col][2] < 1e-7:
        return
    yellow_points[row, col][2] = 0
    pending_draw_yellow_points.append(point)


def add_point_to_red_points(point):
    row = int(point[0] // cell_size)
    col = int(point[1] // cell_size)
    if len(available_ids_red_points[row][col]) == 0:
        print("WARN: No available IDs")
        return
    id_ = available_ids_red_points[row][col].pop()
    red_points_data[row, col, id_, :2] = point
    red_points_data[row, col, id_, 2] = 1
    pending_draw_red_points.append(point)


def add_point_to_blue_points(point):
    blue_queue.append(point)
    if len(blue_queue) >= NUM_BLUE_WAIT:
        point = blue_queue.popleft()
        row = int(point[0] // cell_size)
        col = int(point[1] // cell_size)
        if len(available_ids_blue_points[row][col]) == 0:
            print("WARN: No available IDs")
            return
        id_ = available_ids_blue_points[row][col].pop()
        blue_points_data[row, col, id_, :2] = point
        blue_points_data[row, col, id_, 2] = 1
        pending_draw_blue_points.append(point)


def add_point_to_green_points(point):
    row = int(point[0] // cell_size)
    col = int(point[1] // cell_size)
    if len(available_ids_green_points[row][col]) == 0:
        print("WARN: No available IDs")
        return
    id_ = available_ids_green_points[row][col].pop()
    green_points_data[row, col, id_, :2] = point
    green_points_data[row, col, id_, 2] = 1
    pending_draw_green_points.append(point)


pattern1 = [
    (10, 50),
    (30, 50),
    (50, 50),
    (50, 40),
    (50, 30),
    (50, 20),
    (50, 10),
    (50, 0),
    (50, -25),
    (50, -50),
]
pattern2 = [
    (35, 50),
    (40, 50),
    (50, 50),
    (50, 45),
    (50, 40),
    (50, 35),
    (50, 30),
    (50, 25),
    (50, 20),
    (50, 10),
    (40, 0),
    (45, -10),
    (45, -25),
    (45, -35),
    (50, -50),
]
pattern3 = pattern2[:]


def simulation(delta_time):
    global robot_x, robot_y, robot_orientation
    if delta_time - lidar_full_time > 0:
        delta_time -= lidar_full_time
        for i in range(num_lidar_rays):
            alpha = lidar_angle / 2 - delta_lidar_angle * i + robot_orientation
            point = find_nearest_intersection_jit(
                alpha,
                robot_x + robot_radius * math.cos(robot_orientation),
                robot_y + robot_radius * math.sin(robot_orientation),
                obstacle_segments,
            )
            lidar_rays.append(
                np.array(
                    [
                        [
                            robot_x + robot_radius * math.cos(robot_orientation),
                            robot_y + robot_radius * math.sin(robot_orientation),
                        ],
                        point,
                    ]
                )
            )
            if not check_area_points_jit(point, red_points_data, H, cell_size):
                add_point_to_red_points(point)
    is_blue = check_area_points_jit(
        np.array([robot_x, robot_y]),
        blue_points_data,
        robot_radius * BLUE_RADIUS_FACTOR,
        cell_size,
    )
    is_double_blue = check_area_points_jit(
        np.array([robot_x, robot_y]),
        blue_points_data,
        robot_radius * BLUE_RADIUS_FACTOR * 2,
        cell_size,
    )
    pattern = pattern1
    if is_double_blue:
        pattern = pattern2
    for wL, wR in pattern:
        if is_blue:
            continue
        x, y, orientation = update_robot_position_jit(
            robot_x,
            robot_y,
            robot_orientation,
            wL,
            wR,
            delta_time,
            wheel_diameter,
            wheel_distance / 2,
        )
        if not check_area_points_jit(
            np.array([x, y]),
            red_points_data,
            robot_radius * RED_RADIUS_FACTOR,
            cell_size,
        ):
            if not check_area_points_jit(
                np.array([x, y]),
                blue_points_data,
                robot_radius * BLUE_RADIUS_FACTOR,
                cell_size,
            ):
                robot_x, robot_y, robot_orientation = x, y, orientation
                print(wL, wR)
                break
    else:
        for wL, wR in [(random.uniform(20, 50), random.uniform(20, 50))] + pattern3:
            x, y, orientation = update_robot_position_jit(
                robot_x,
                robot_y,
                robot_orientation,
                wL,
                wR,
                delta_time,
                wheel_diameter,
                wheel_distance / 2,
            )
            if not check_area_points_jit(
                np.array([x, y]),
                red_points_data,
                robot_radius * RED_RADIUS_FACTOR,
                cell_size,
            ):
                robot_x, robot_y, robot_orientation = x, y, orientation
                print(wL, wR)
                break
        else:
            robot_x, robot_y, robot_orientation = update_robot_position_jit(
                robot_x,
                robot_y,
                robot_orientation,
                50,
                -50,
                delta_time,
                wheel_diameter,
                wheel_distance / 2,
            )
    point = np.array([robot_x, robot_y])
    y_tmp = check_cleared_point_jit(robot_x, robot_y, yellow_points, robot_radius)
    if y_tmp is not None:
        add_point_to_yellow_points(y_tmp)
    if not check_area_points_jit(point, blue_points_data, H * 3, cell_size):
        add_point_to_blue_points(point)
        add_point_to_green_points(point)


pygame.init()
clock = pygame.time.Clock()
screen = pygame.display.set_mode((2 * K * room_width, 2 * K * room_height))
pygame.display.set_caption("Moving the Robot Vacuum Cleaner in the Room")
obstacle_surface = pygame.surface.Surface(
    (K * room_width, K * room_height), pygame.SRCALPHA
)
yellow_points_surface = pygame.surface.Surface(
    (K * room_width, K * room_height), pygame.SRCALPHA
)
for i in range(num_rows):
    for j in range(num_cols):
        if yellow_points[i, j, 2] > 0.0:
            pygame.draw.circle(
                yellow_points_surface, YELLOW, yellow_points[i, j, :2] * K, 0.1 * K
            )
            continue
        if yellow_points[i, j, 2] < 0.0:
            pygame.draw.circle(
                yellow_points_surface, "red", yellow_points[i, j, :2] * K, 0.1 * K
            )
            continue
red_points_surface = pygame.surface.Surface(
    (K * room_width, K * room_height), pygame.SRCALPHA
)
blue_point_surface = pygame.surface.Surface(
    (K * room_width, K * room_height), pygame.SRCALPHA
)
double_blue_point_surface = pygame.surface.Surface(
    (K * room_width, K * room_height), pygame.SRCALPHA
)
green_point_surface = pygame.surface.Surface(
    (K * room_width, K * room_height), pygame.SRCALPHA
)
trajectory_surface = pygame.surface.Surface(
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
    # t = 1 / clock.get_fps() if clock.get_fps() != 0 else 0
    t = moving_full_time + lidar_full_time
    simulation(t)
    T += t
    timer_trajectory -= t
    if timer_trajectory <= 0:
        pygame.draw.line(
            trajectory_surface,
            (167, 25, 220, 128),
            last_point * K,
            np.array([robot_x, robot_y]) * K,
            2,
        )
        last_point = np.array([robot_x, robot_y])
    while len(pending_draw_yellow_points) != 0:
        yellow_point = pending_draw_yellow_points.pop()
        pygame.draw.circle(
            yellow_points_surface,
            "red",
            yellow_point * K,
            1,
        )

    while len(pending_draw_red_points) != 0:
        red_point = pending_draw_red_points.pop()
        pygame.draw.circle(
            red_points_surface,
            (255, 0, 0, 128),
            red_point * K,
            robot_radius * K * RED_RADIUS_FACTOR,
        )

    while len(pending_draw_blue_points) != 0:
        blue_point = pending_draw_blue_points.pop()
        pygame.draw.circle(
            double_blue_point_surface,
            (0, 0, 255, 60),
            blue_point * K,
            robot_radius * K * BLUE_RADIUS_FACTOR * 2,
        )
        pygame.draw.circle(
            blue_point_surface,
            (0, 0, 255, 128),
            blue_point * K,
            robot_radius * K * BLUE_RADIUS_FACTOR,
        )

    while len(pending_draw_green_points) != 0:
        green_point = pending_draw_green_points.pop()
        pygame.draw.circle(
            green_point_surface,
            (0, 255, 0, 128),
            green_point * K,
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

    red_surface = main_surface.copy()
    red_surface.blit(red_points_surface, (0, 0))
    for line in lidar_rays:
        pygame.draw.line(red_surface, "red", line[0] * K, line[1] * K, 1)

    blue_surface = main_surface.copy()
    blue_surface.blit(double_blue_point_surface, (0, 0))
    blue_surface.blit(blue_point_surface, (0, 0))

    green_surface = main_surface.copy()
    green_surface.blit(green_point_surface, (0, 0))
    green_surface.blit(yellow_points_surface, (0, 0))
    main_surface.blit(trajectory_surface, (0, 0))

    screen.blit(main_surface, (0, 0))
    screen.blit(red_surface, (K * room_width, 0))
    screen.blit(blue_surface, (0, K * room_height))
    screen.blit(green_surface, (K * room_width, K * room_height))
    pygame.display.update()
    clock.tick(1000)
