import math
import random
from collections import deque
import numpy as np
import pygame
from functions import (
    update_robot_position_jit,
    find_nearest_intersection_jit,
    check_red_points_jit,
    closest_yellow_point_jit,
    initialize_yellow_points_jit,
    closest_red_point_angle,
    create_trajectory,
    calculate_yellow_point_closest_jit,
)

H = 0.01
LIDAR_ROTATION_TIME = 2.6527945914557116e-5
LIDAR_TIME = 6.7e-8
RED_RADIUS_FACTOR = 1.2
YELLOW_RADIUS_FACTOR = 0.8
TRAJECTORY_RADIUS_FACTOR = 0.2
TIMER_TRAJECTORY = 1
MAX_W = 10
MAX_DW = 1
YELLOW = "#a09a07"

print("All data is measured in meters (or radians)!!!")
room_width, room_height = tuple(map(float, input("Room size:").split()))
K = 400 / room_height
print("---Characteristics of the robot vacuum cleaner---")
robot_radius = float(input("The radius of robot:"))
wheel_diameter = float(input("The diameter of wheels:"))
wheel_distance = float(input("The distance between the wheels:"))
lidar_angle = float(input("The laser locator angle (radian):"))
DELTA_LIDAR_ANGLE = 1 / 30
robot_x, robot_y = tuple(map(float, input("The start position of robot:").split()))
robot_orientation = 0
cell_size = 2 * robot_radius
yellow_cell_size = robot_radius / 1.2
num_rows = int(room_width // cell_size + 2)
num_cols = int(room_height // cell_size + 2)
yellow_num_rows = int(room_width // yellow_cell_size + 1)
yellow_num_cols = int(room_height // yellow_cell_size + 1)


print("---Obstacles---")
num_obstacles = int(input("Number of obstacles:"))
obstacle_vertices = []
total_segments = 4

num_lidar_rays = int((1 / DELTA_LIDAR_ANGLE) * lidar_angle)

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

non_initialized_ids = set()
yellow_points = np.zeros((yellow_num_rows, yellow_num_cols, 3), dtype=np.float64)
for i in range(yellow_num_rows):
    for j in range(yellow_num_cols):
        non_initialized_ids.add((i, j))
        yellow_points[i][j] = np.array(
            [
                yellow_cell_size / 2 + i * yellow_cell_size,
                yellow_cell_size / 2 + j * yellow_cell_size,
                -1.0,
            ],
            dtype=np.float64,
        )

red_points_data = np.zeros((num_rows, num_cols, 1000, 3), dtype=float)
green_points_data = np.zeros((num_rows, num_cols, 1000, 3), dtype=float)
available_ids_red_points = [
    [set(range(1000)) for _ in range(num_cols)] for _ in range(num_rows)
]
available_ids_blue_points = [
    [set(range(1000)) for _ in range(num_cols)] for _ in range(num_rows)
]
available_ids_green_points = [
    [set(range(1000)) for _ in range(num_cols)] for _ in range(num_rows)
]
pending_draw_yellow_points = deque()
pending_draw_initialized_yellow_points = deque()
pending_draw_red_points = deque()
pending_draw_green_points = deque()
lidar_full_time = (LIDAR_TIME + LIDAR_ROTATION_TIME) * num_lidar_rays
moving_full_time = 1e-2
lidar_rays = []
T = 0
timer_trajectory = TIMER_TRAJECTORY
last_point = np.array([robot_x, robot_y])


def add_point_to_yellow_points(point):
    row = int(point[0] // yellow_cell_size)
    col = int(point[1] // yellow_cell_size)
    if yellow_points[row, col][2] < 1e-7:
        return
    yellow_points[row, col][2] = 0
    pending_draw_yellow_points.append(point)


def add_point_to_initialized_yellow_points(point):
    row = int(point[0] // yellow_cell_size)
    col = int(point[1] // yellow_cell_size)
    non_initialized_ids.discard((row, col))
    yellow_points[row][col][2] = 1.0
    pending_draw_initialized_yellow_points.append(point)


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


last_kL, last_kR = 0.0, 0.0
dk = MAX_DW / MAX_W
trajectory = None


def simulation(delta_time):
    global stop_count
    global robot_x, robot_y, robot_orientation
    global last_kR, last_kL
    global save_time
    global trajectory

    if delta_time - lidar_full_time > 0:
        delta_time -= lidar_full_time

        pos_x = robot_x + robot_radius * math.cos(robot_orientation)
        pos_y = robot_y + robot_radius * math.sin(robot_orientation)

        for i in range(num_lidar_rays):
            alpha = lidar_angle / 2 - DELTA_LIDAR_ANGLE * i + robot_orientation
            point = find_nearest_intersection_jit(
                alpha, pos_x, pos_y, obstacle_segments
            )
            lidar_rays.append(np.array([[pos_x, pos_y], point]))

            if not check_red_points_jit(point, red_points_data, H, cell_size):
                add_point_to_red_points(point)

            if (
                yellow_point := closest_yellow_point_jit(
                    point[0], point[1], yellow_points, robot_radius * RED_RADIUS_FACTOR
                )
            ) is not None:
                add_point_to_yellow_points(yellow_point)
    random.shuffle(lidar_rays)
    if len(non_initialized_ids) != 0:
        for lidar_segment in lidar_rays[:5]:
            for initialized_yellow_point in initialize_yellow_points_jit(
                lidar_segment, yellow_points, robot_radius / 2, non_initialized_ids
            ):
                add_point_to_initialized_yellow_points(initialized_yellow_point)
    point = calculate_yellow_point_closest_jit(robot_x, robot_y, yellow_points)
    if point is None:
        return
    if trajectory is None:
        trajectory = create_trajectory(
            robot_x,
            robot_y,
            point[0],
            point[1],
            red_points_data,
            robot_radius * RED_RADIUS_FACTOR * 1.001,
            cell_size,
            0.3,
            math.radians(5),
            1,
        )
        if trajectory is None:
            trajectory = create_trajectory(
                robot_x,
                robot_y,
                point[0],
                point[1],
                red_points_data,
                robot_radius * RED_RADIUS_FACTOR * 1.001,
                cell_size,
                0.3,
                math.radians(5),
                -1,
            )
            if trajectory is None:
                add_point_to_yellow_points(point)
    if trajectory is None or len(trajectory) == 0:
        trajectory = None
        need_point = point
    else:
        need_point = trajectory[0]
    if check_red_points_jit(
        need_point, red_points_data, robot_radius * RED_RADIUS_FACTOR, cell_size
    ):
        trajectory = None
        need_point = point
    angle = np.arctan2(need_point[1] - robot_y, need_point[0] - robot_x)
    global help_yellow_angle, start_angle, yellow_angle
    help_yellow_angle = closest_red_point_angle(
        np.array([robot_x, robot_y]),
        red_points_data,
        cell_size,
        robot_radius * RED_RADIUS_FACTOR * 1.1,
    )
    yellow_angle = angle
    delta = (angle - robot_orientation + math.pi) % (2 * math.pi) - math.pi
    if delta > 0:
        nL = 1 - (3 / math.pi) * delta
        nR = 1

    else:
        nL = 1
        nR = 1 - (3 / math.pi) * (-delta)
    if last_kL >= nL:
        kL = last_kL - dk if last_kL - dk > nL else nL
    else:
        kL = last_kL + dk if last_kL + dk < nL else nL
    if last_kR >= nR:
        kR = last_kR - dk if last_kR - dk > nR else nR
    else:
        kR = last_kR + dk if last_kR - dk < nR else nR

    x, y, orientation = update_robot_position_jit(
        robot_x,
        robot_y,
        robot_orientation,
        kL,
        kR,
        delta_time,
        wheel_diameter,
        wheel_distance / 2,
        MAX_W,
    )
    if not check_red_points_jit(
        np.array([x, y]),
        red_points_data,
        robot_radius * RED_RADIUS_FACTOR,
        cell_size,
    ):
        last_kL, last_kR = kL, kR
        print(last_kL * MAX_W, last_kR * MAX_W)
        robot_x, robot_y, robot_orientation = x, y, orientation
    else:
        diff = (yellow_angle - help_yellow_angle) % (2 * math.pi)
        if diff > np.pi:
            diff -= 2 * np.pi
        new_angle = help_yellow_angle + diff / 2
        delta = (new_angle - robot_orientation + math.pi) % (2 * math.pi) - math.pi
        if delta > 0:
            nL = 1 - (3 / math.pi) * delta
            nR = 1

        else:
            nL = 1
            nR = 1 - (3 / math.pi) * (-delta)
        if last_kL > nL:
            kL = last_kL - dk if last_kL - dk > nL else nL
        else:
            kL = last_kL + dk if last_kL + dk < nL else nL
        if last_kR > nR:
            kR = last_kR - dk if last_kR - dk > nR else nR
        else:
            kR = last_kR + dk if last_kR - dk < nR else nR

        x, y, orientation = update_robot_position_jit(
            robot_x,
            robot_y,
            robot_orientation,
            kL,
            kR,
            delta_time,
            wheel_diameter,
            wheel_distance / 2,
            MAX_W,
        )
        if not check_red_points_jit(
            np.array([x, y]),
            red_points_data,
            robot_radius * RED_RADIUS_FACTOR,
            cell_size,
        ):
            last_kL, last_kR = kL, kR
            print(last_kL * MAX_W, last_kR * MAX_W)
            robot_x, robot_y, robot_orientation = x, y, orientation
        else:
            new_angle = help_yellow_angle
            delta = (new_angle - robot_orientation + math.pi) % (2 * math.pi) - math.pi
            if delta > 0:
                nL = 1 - (3 / math.pi) * delta
                nR = 1

            else:
                nL = 1
                nR = 1 - (3 / math.pi) * (-delta)
            if last_kL > nL:
                kL = last_kL - dk if last_kL - dk > nL else nL
            else:
                kL = last_kL + dk if last_kL + dk < nL else nL
            if last_kR > nR:
                kR = last_kR - dk if last_kR - dk > nR else nR
            else:
                kR = last_kR + dk if last_kR - dk < nR else nR

            x, y, orientation = update_robot_position_jit(
                robot_x,
                robot_y,
                robot_orientation,
                kL,
                kR,
                delta_time,
                wheel_diameter,
                wheel_distance / 2,
                MAX_W,
            )
            if not check_red_points_jit(
                np.array([x, y]),
                red_points_data,
                robot_radius * RED_RADIUS_FACTOR,
                cell_size,
            ):
                last_kL, last_kR = kL, kR
                print(last_kL * MAX_W, last_kR * MAX_W)
                robot_x, robot_y, robot_orientation = x, y, orientation
            else:
                k_ = (abs(last_kR) + abs(last_kL)) / 2
                if delta > 0:
                    kL = -k_
                    kR = k_
                else:
                    kL = k_
                    kR = -k_
                x, y, orientation = update_robot_position_jit(
                    robot_x,
                    robot_y,
                    robot_orientation,
                    kL,
                    kR,
                    delta_time,
                    wheel_diameter,
                    wheel_distance / 2,
                    MAX_W,
                )
                if not check_red_points_jit(
                    np.array([x, y]),
                    red_points_data,
                    robot_radius * RED_RADIUS_FACTOR,
                    cell_size,
                ):
                    last_kL, last_kR = kL, kR
                    print(last_kL * MAX_W, last_kR * MAX_W)
                    robot_x, robot_y, robot_orientation = x, y, orientation
    point = np.array([robot_x, robot_y])
    y_tmp = closest_yellow_point_jit(
        robot_x, robot_y, yellow_points, robot_radius * YELLOW_RADIUS_FACTOR
    )
    if (
        abs(need_point[0] - robot_x) < robot_radius * TRAJECTORY_RADIUS_FACTOR
        and abs(need_point[1] - robot_y) < robot_radius * TRAJECTORY_RADIUS_FACTOR
    ):
        if trajectory is None or len(trajectory) == 0:
            trajectory = None
        else:
            trajectory = trajectory[1:]
    if y_tmp is not None:
        add_point_to_yellow_points(y_tmp)

    if not check_red_points_jit(point, green_points_data, H, cell_size):
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
for i in range(yellow_num_rows):
    for j in range(yellow_num_cols):
        if yellow_points[i, j, 2] > 0.0:
            pygame.draw.circle(
                yellow_points_surface, YELLOW, yellow_points[i, j, :2] * K, 1
            )
            continue
        if yellow_points[i, j, 2] < 0.0:
            pygame.draw.circle(
                yellow_points_surface, "black", yellow_points[i, j, :2] * K, 1
            )
            continue
red_points_surface = pygame.surface.Surface(
    (K * room_width, K * room_height), pygame.SRCALPHA
)
green_point_surface = pygame.surface.Surface(
    (K * room_width, K * room_height), pygame.SRCALPHA
)
trajectory_surface = pygame.surface.Surface(
    (K * room_width, K * room_height), pygame.SRCALPHA
)
yellow_angle = None
help_yellow_angle = None
for i in range(total_segments):
    pygame.draw.line(
        obstacle_surface,
        "black",
        obstacle_segments[i, 0, :] * K,
        obstacle_segments[i, 1, :] * K,
    )
save_time = 0
start_angle = None
stop_count = 0
while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
    lidar_rays = []
    t = moving_full_time + lidar_full_time
    yellow_angle = None
    old_x, old_y = robot_x, robot_y
    simulation(t)
    if abs(old_x - robot_x) < 1e-5 and abs(old_y - robot_y) < 1e-5:
        stop_count += 1
        if stop_count > 10:
            save_time = 20
            stop_count = 0
            start_angle = None
    else:
        stop_count = 0

    save_time = max(save_time - t, 0)
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
    while len(pending_draw_initialized_yellow_points) != 0:
        yellow_point = pending_draw_initialized_yellow_points.pop()
        pygame.draw.circle(
            yellow_points_surface,
            YELLOW,
            yellow_point * K,
            1,
        )

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

    yellow_surface = main_surface.copy()
    yellow_surface.blit(yellow_points_surface, (0, 0))

    green_surface = main_surface.copy()
    green_surface.blit(green_point_surface, (0, 0))
    if yellow_angle is not None:
        pygame.draw.line(
            yellow_surface,
            "yellow",
            np.array([robot_x, robot_y]) * K,
            np.array(
                [
                    robot_x + 100 * math.cos(yellow_angle),
                    robot_y + 100 * math.sin(yellow_angle),
                ]
            )
            * K,
            3,
        )
    if help_yellow_angle is not None:
        pygame.draw.line(
            yellow_surface,
            "red",
            np.array([robot_x, robot_y]) * K,
            np.array(
                [
                    robot_x + 100 * math.cos(help_yellow_angle),
                    robot_y + 100 * math.sin(help_yellow_angle),
                ]
            )
            * K,
            1,
        )
    main_surface.blit(trajectory_surface, (0, 0))
    if trajectory is not None and len(trajectory) > 0:
        pygame.draw.line(
            main_surface, "red", np.array([robot_x, robot_y]) * K, trajectory[0] * K, 1
        )
        for i, point in enumerate(trajectory[:-1]):
            pygame.draw.line(main_surface, "red", point * K, trajectory[i + 1] * K, 1)
    screen.blit(main_surface, (0, 0))
    screen.blit(red_surface, (K * room_width, 0))
    screen.blit(yellow_surface, (0, K * room_height))
    screen.blit(green_surface, (K * room_width, K * room_height))
    pygame.display.update()
    clock.tick(1200)
