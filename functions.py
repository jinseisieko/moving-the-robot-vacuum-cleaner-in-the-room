import math
import numba as nb
import numpy as np


@nb.njit()
def new_values(x, y, alpha, wL, wR, dt, D, K):
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
    ray_end = np.array([math.cos(angle) * 10e7, math.sin(angle) * 10e7], dtype=float)
    ray_segment = np.array([[start_x, start_y], ray_end], dtype=float)
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
