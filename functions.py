import math
import numba as nb
import numpy as np


@nb.njit()
def update_robot_position_jit(x, y, alpha, wL, wR, dt, D, K):
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


def find_nearest_intersection(angle, start_x, start_y, segments):
    ray_end = np.array(
        [math.cos(angle) * 10e7, math.sin(angle) * 10e7], dtype=np.float64
    )
    ray_segment = np.array([[start_x, start_y], ray_end], dtype=np.float64)
    segments_copy = segments.copy()
    segment_start_points = segments_copy[:, 0]
    segment_end_points = segments_copy[:, 1]
    direction_segment = segment_end_points - segment_start_points
    direction_ray = ray_segment[1] - ray_segment[0]
    matrix = np.zeros((len(segments_copy), 2, 2))
    matrix[:, 0, 0] = direction_segment[:, 0]
    matrix[:, 1, 0] = direction_segment[:, 1]
    matrix[:, 0, 1] = direction_ray[0]
    matrix[:, 1, 1] = direction_ray[1]
    determinant = np.linalg.det(matrix)
    non_zero_determinant = ~np.isclose(determinant, 0)
    t = np.cross(ray_segment[0] - segment_start_points, direction_ray) / determinant
    u = np.cross(ray_segment[0] - segment_start_points, direction_segment) / determinant
    is_t_in_range = (0 <= t) & (t <= 1)
    is_u_in_range = (0 <= u) & (u <= 1)
    intersection_points = segment_start_points + t[:, np.newaxis] * direction_segment
    if (
        len(intersection_points[is_t_in_range & is_u_in_range & non_zero_determinant])
        == 0
    ):
        return None
    valid_intersections = intersection_points[
        is_t_in_range & is_u_in_range & non_zero_determinant
    ].copy()
    distances = np.linalg.norm(
        valid_intersections - np.array([start_x, start_y]), axis=1
    )
    return valid_intersections[np.argmin(distances)]


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


def check_area_points(point, warning_points, radius, cell_size):
    row = int(point[0] // cell_size)
    col = int(point[1] // cell_size)
    warning_points_copy = warning_points.copy()
    warning_points_copy[row, col, :, 0:2] -= point
    warning_points_copy[row, col, :, 0:2] **= 2
    warning_points_copy[row, col, :, 0] = np.sqrt(
        warning_points_copy[row, col, :, 0] + warning_points_copy[row, col, :, 1]
    )
    is_in_area = any(
        (warning_points_copy[row, col, :, 0] < radius)
        & (warning_points_copy[row, col, :, 2] > 0.0)
    )
    return is_in_area


@nb.njit
def check_area_points_jit(point, warning_points, radius, cell_size):
    row = int(point[0] // cell_size)
    col = int(point[1] // cell_size)
    is_in_area = False
    for k in [-1, 0, 1]:
        for j in [-1, 0, 1]:
            warning_points_copy = warning_points[
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
def check_cleared_point_jit(x, y, yellow_points, radius):
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
def calculate_average_angle_jit(robot_x, robot_y, yellow_points):
    angles = []
    for i in range(yellow_points.shape[0]):
        for j in range(yellow_points.shape[1]):
            if yellow_points[i][j][2] > 0:
                point_x = yellow_points[i][j][0]
                point_y = yellow_points[i][j][1]
                angle = np.arctan2(point_y - robot_y, point_x - robot_x)
                angles.append(angle)
    angles = np.array(angles)
    min_sum = float('inf')
    optimal_angle = None
    for angle in angles:
        sum_diffs = np.sum(np.abs(angles - angle))

        if sum_diffs < min_sum:
            min_sum = sum_diffs
            optimal_angle = angle
    return optimal_angle

@nb.njit
def initialize_yellow_points(segment, yellow_points, radius, non_initialized_ids):
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
            if distance_squared <= radius ** 2:
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

        if distance_squared <= radius ** 2:
            initialized_points.append(point)

    return initialized_points


