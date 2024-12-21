import math
import random

import numba as nb
import numpy as np


@nb.njit()
def update_robot_position_jit(x, y, alpha, kL, kR, dt, D, K, max_w):
    wL, wR = max_w * kL, max_w * kR
    if abs(wL - wR) < 10e-10:
        new_alpha = alpha
        new_x = x + dt * (wL * D / 2) * math.cos(alpha)
        new_y = y + dt * (wL * D / 2) * math.sin(alpha)
    else:
        new_alpha = alpha + ((D * (wR - wL) * dt) / (4 * K))
        new_x = x + (
            ((wR + wL) / (wR - wL)) * K * (math.sin(new_alpha) - math.sin(alpha))
        )
        new_y = y + (
            (-(wR + wL) / (wR - wL)) * K * (math.cos(new_alpha) - math.cos(alpha))
        )
    return new_x, new_y, new_alpha


@nb.njit
def find_nearest_intersection_jit(angle, start_x, start_y, segments):
    ray_end_x = math.cos(angle) * 10e7
    ray_end_y = math.sin(angle) * 10e7
    ray_segment = np.array(
        [[start_x, start_y], [ray_end_x, ray_end_y]], dtype=np.float64
    )
    num_segments = segments.shape[0]
    segment_start_points = segments[:, 0]
    segment_end_points = segments[:, 1]
    direction_segment = segment_end_points - segment_start_points
    direction_ray = ray_segment[1] - ray_segment[0]
    min_distance = float("inf")
    nearest_intersection = None
    for i in range(num_segments):
        det = (
            direction_segment[i, 0] * direction_ray[1]
            - direction_segment[i, 1] * direction_ray[0]
        )
        if det != 0:
            p = ray_segment[0] - segment_start_points[i]
            t = (p[0] * direction_ray[1] - p[1] * direction_ray[0]) / det
            u = (p[0] * direction_segment[i, 1] - p[1] * direction_segment[i, 0]) / det
            if 0 <= t <= 1 and 0 <= u <= 1:
                intersection_x = (
                    segment_start_points[i, 0] + t * direction_segment[i, 0]
                )
                intersection_y = (
                    segment_start_points[i, 1] + t * direction_segment[i, 1]
                )
                distance = math.sqrt(
                    (intersection_x - start_x) ** 2 + (intersection_y - start_y) ** 2
                )
                if distance < min_distance:
                    min_distance = distance
                    nearest_intersection = np.array([intersection_x, intersection_y])
    return nearest_intersection


@nb.njit
def check_red_points_jit(point, red_points, radius, cell_size):
    row = int(point[0] // cell_size)
    col = int(point[1] // cell_size)
    is_in_area = False
    for k in [-1, 0, 1]:
        for j in [-1, 0, 1]:
            warning_points_copy = red_points[
                max(0, row + k), max(0, col + j), :, :
            ].copy()
            warning_points_copy[:, 0] -= point[0]
            warning_points_copy[:, 1] -= point[1]
            distances_squared = (
                warning_points_copy[:, 0] ** 2 + warning_points_copy[:, 1] ** 2
            )
            distances = np.sqrt(distances_squared)
            for i in range(warning_points_copy.shape[0]):
                if distances[i] < radius and warning_points_copy[i, 2] > 0.0:
                    is_in_area = True
                    break
    return is_in_area


@nb.njit
def count_intersections_jit(start_x, start_y, end_x, end_y, segments):
    ray_segment = np.array([[start_x, start_y], [end_x, end_y]], dtype=np.float64)
    num_segments = segments.shape[0]
    segment_start_points = segments[:, 0]
    segment_end_points = segments[:, 1]
    direction_segment = segment_end_points - segment_start_points
    direction_ray = ray_segment[1] - ray_segment[0]
    intersection_count = 0
    for i in range(num_segments):
        det = (
            direction_segment[i, 0] * direction_ray[1]
            - direction_segment[i, 1] * direction_ray[0]
        )
        if det != 0:
            p = ray_segment[0] - segment_start_points[i]
            t = (p[0] * direction_ray[1] - p[1] * direction_ray[0]) / det
            u = (p[0] * direction_segment[i, 1] - p[1] * direction_segment[i, 0]) / det
            if 0 <= t <= 1 and 0 <= u <= 1:
                intersection_count += 1
    return intersection_count


@nb.njit
def closest_yellow_point_jit(x, y, yellow_points, radius):
    closest_point = None
    min_distance_squared = float("inf")
    for i in range(yellow_points.shape[0]):
        for j in range(yellow_points.shape[1]):
            if yellow_points[i][j][2] > 0:
                point_x = yellow_points[i][j][0]
                point_y = yellow_points[i][j][1]
                dx = point_x - x
                dy = point_y - y
                distance_squared = dx * dx + dy * dy
                if distance_squared <= radius * radius:
                    if distance_squared < min_distance_squared:
                        min_distance_squared = distance_squared
                        closest_point = np.array([point_x, point_y])
    return closest_point


@nb.njit
def calculate_yellow_angle_closest_jit(robot_x, robot_y, yellow_points):
    min_distance = float("inf")
    priority = -1
    optimal_angle = None
    for i in range(yellow_points.shape[0]):
        for j in range(yellow_points.shape[1]):
            if yellow_points[i][j][2] > 0:
                point_x = yellow_points[i][j][0]
                point_y = yellow_points[i][j][1]
                dx = point_x - robot_x
                dy = point_y - robot_y
                distance_squared = dx * dx + dy * dy
                angle = np.arctan2(point_y - robot_y, point_x - robot_x)
                if distance_squared < min_distance:
                    min_distance = distance_squared
                    optimal_angle = angle
                    priority = yellow_points[i][j][2]
    return optimal_angle


@nb.njit
def calculate_yellow_angle_smallest_jit(
    robot_x, robot_y, robot_orientation, yellow_points
):
    min_arc = float("inf")
    optimal_angle = None
    for i in range(yellow_points.shape[0]):
        for j in range(yellow_points.shape[1]):
            if yellow_points[i][j][2] > 0:
                point_x = yellow_points[i][j][0]
                point_y = yellow_points[i][j][1]
                angle = np.arctan2(point_y - robot_y, point_x - robot_x)
                delta = abs(angle % (2 * math.pi) - robot_orientation % (2 * math.pi))
                arc = min(delta, 2 * math.pi - delta)
                if arc < min_arc:
                    min_arc = arc
                    optimal_angle = angle
    return optimal_angle


@nb.njit
def initialize_yellow_points_jit(segment, yellow_points, radius, non_initialized_ids):
    initialized_points = []
    A = segment[0]
    B = segment[1]
    AB = B - A
    AB_length_squared = np.dot(AB, AB)

    for i, j in non_initialized_ids:
        point = yellow_points[i][j][:2]
        C = point
        AC = C - A

        if AB_length_squared == 0:
            distance_squared = np.dot(AC, AC)
            if distance_squared <= radius**2:
                initialized_points.append(point)
            continue

        t = np.dot(AC, AB) / AB_length_squared

        if t < 0:
            nearest_point = A
        elif t > 1:
            nearest_point = B
        else:
            nearest_point = A + t * AB

        distance_squared = np.dot(C - nearest_point, C - nearest_point)

        if distance_squared <= radius**2:
            initialized_points.append(point)

    return initialized_points


def update_priority_yellow_points(x, y, yellow_points, radius):
    for i in range(yellow_points.shape[0]):
        for j in range(yellow_points.shape[1]):
            if yellow_points[i][j][2] > 0:
                point_x = yellow_points[i][j][0]
                point_y = yellow_points[i][j][1]
                dx = point_x - x
                dy = point_y - y
                distance_squared = dx * dx + dy * dy
                if distance_squared <= radius * radius:
                    yellow_points[i][j][2] += 0.1


@nb.njit
def closest_red_point_angle(point, red_points, cell_size, radius):
    row = int(point[0] // cell_size)
    col = int(point[1] // cell_size)
    closest_point = None
    distance = float("inf")
    for k in [-1, 0, 1]:
        for j in [-1, 0, 1]:
            warning_points_copy = red_points[max(0, row + k), max(0, col + j), :, :]
            valid_points = warning_points_copy[warning_points_copy[:, 2] > 0]

            if valid_points.shape[0] == 0:
                continue
            dx = valid_points[:, 0] - point[0]
            dy = valid_points[:, 1] - point[1]
            distances_squared = dx**2 + dy**2
            min_index = np.argmin(distances_squared)
            closest_candidate = valid_points[min_index]
            closest_distance = np.sqrt(distances_squared[min_index])
            if closest_distance < distance:
                closest_point = closest_candidate
                distance = closest_distance

    if closest_point is None:
        return None
    if distance > radius:
        return None
    angle = (
        np.arctan2(closest_point[1] - point[1], closest_point[0] - point[0]) + math.pi
    )
    return angle


@nb.njit
def create_trajectory(
    x1, y1, x2, y2, red_points, radius, cell_size, delta_s, delta_a, k=None
):
    max_points = 10000
    points = np.empty((max_points, 2))
    count = 0
    k = random.randint(0, 1) * 2 - 1 if k is None else k

    while x1 != x2 and y1 != y2:
        p = min(delta_s, ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5)
        angle = math.atan2(y2 - y1, x2 - x1)
        next_x = x1 + p * math.cos(angle)
        next_y = y1 + p * math.sin(angle)
        flag = True
        count_trying = 0
        while (
            check_red_points_jit(
                np.array([next_x, next_y]), red_points, radius, cell_size
            )
            or flag
        ):
            count_trying += 1
            if count_trying > 361:
                return None
            next_x = x1 + p * math.cos(angle)
            next_y = y1 + p * math.sin(angle)
            angle += k * delta_a
            flag = False

            for i in range(max(0, count - 30), count - 1):
                A = points[i]
                B = points[i + 1]
                AB = B - A
                AB_length_squared = np.dot(AB, AB)
                C = np.array([next_x, next_y])
                AC = C - A

                if AB_length_squared == 0:
                    distance_squared = np.dot(AC, AC)
                    if distance_squared <= (delta_s / 2) ** 2:
                        flag = True
                    continue

                t = np.dot(AC, AB) / AB_length_squared
                if t < 0:
                    nearest_point = A
                elif t > 1:
                    nearest_point = B
                else:
                    nearest_point = A + t * AB

                distance_squared = np.dot(C - nearest_point, C - nearest_point)
                if distance_squared <= (delta_s / 2) ** 2:
                    flag = True

            if p < delta_s:
                flag = False
        if count > max_points:
            return None

        if count_trying < 1:
            count -= 1
            dx = points[count - 1, 0] - next_x
            dy = points[count - 1, 1] - next_y
            if dx**2 + dy**2 > (delta_s**2) * 5:
                count += 1
        points[count, 0] = next_x
        points[count, 1] = next_y
        count += 1

        x1 = next_x
        y1 = next_y

    return points[:count]


@nb.njit
def calculate_yellow_point_closest_jit(robot_x, robot_y, yellow_points):
    min_distance = float("inf")
    point = None
    for i in range(yellow_points.shape[0]):
        for j in range(yellow_points.shape[1]):
            if yellow_points[i][j][2] > 0:
                point_x = yellow_points[i][j][0]
                point_y = yellow_points[i][j][1]
                dx = point_x - robot_x
                dy = point_y - robot_y
                distance_squared = dx * dx + dy * dy
                if distance_squared < min_distance:
                    point = yellow_points[i][j][:2]
                    min_distance = distance_squared
    return point
