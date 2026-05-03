from __future__ import annotations

import numpy as np


def generate_initial_conditions_homogeneous(eq: np.ndarray, buffer: float, num_samples: int) -> np.ndarray:
    min_x1 = -buffer + eq[0]
    max_x1 = buffer + eq[0]
    min_x2 = -buffer + eq[1]
    max_x2 = buffer + eq[1]

    offset = buffer / num_samples
    x1 = np.vstack((np.linspace(min_x1, max_x1 - offset, num_samples), min_x2 * np.ones(num_samples)))
    x2 = np.vstack((max_x1 * np.ones(num_samples), np.linspace(min_x2, max_x2 - offset, num_samples)))
    x3 = np.vstack((np.linspace(min_x1 + offset, max_x1, num_samples), max_x2 * np.ones(num_samples)))
    x4 = np.vstack((min_x1 * np.ones(num_samples), np.linspace(min_x2 + offset, max_x2, num_samples)))
    return np.hstack((x1, x2, x3, x4))


def generate_initial_conditions_dense_margins(eq: np.ndarray, buffer: float, num_samples: int) -> np.ndarray:
    min_x1 = -buffer + eq[0]
    max_x1 = buffer + eq[0]
    min_x2 = -buffer + eq[1]
    max_x2 = buffer + eq[1]

    offset = buffer / num_samples
    multiple = 3
    regions = 2 * multiple + 1
    x1_part1 = np.linspace(min_x1, -1.0, multiple * (num_samples // (regions)), endpoint=False)
    x1_part2 = np.linspace(-1.0, 1.0, num_samples - 2 * multiple * (num_samples // (regions)), endpoint=False)
    x1_part3 = np.linspace(1.0, max_x1, multiple * (num_samples // (regions)))
    x1_full = np.concatenate((x1_part1, x1_part2, x1_part3))

    x1 = np.vstack((x1_full, min_x2 * np.ones(num_samples)))
    x2 = np.vstack((max_x1 * np.ones(num_samples), np.linspace(min_x2, max_x2 - offset, num_samples)))
    x3 = np.vstack((x1_full[::-1], max_x2 * np.ones(num_samples)))
    x4 = np.vstack((min_x1 * np.ones(num_samples), np.linspace(min_x2 + offset, max_x2, num_samples)))
    return np.hstack((x1, x2, x3, x4))


def generate_initial_conditions_multiple_density(eq: np.ndarray, buffer: float, num_samples: int) -> np.ndarray:
    min_x1 = -buffer + eq[0]
    max_x1 = buffer + eq[0]
    min_x2 = -buffer + eq[1]
    max_x2 = buffer + eq[1]

    offset = buffer / num_samples
    multiple = 10
    x1 = np.vstack((np.linspace(min_x1, max_x1 - offset / multiple, num_samples * multiple), min_x2 * np.ones(num_samples * multiple)))
    x2 = np.vstack((max_x1 * np.ones(num_samples), np.linspace(min_x2, max_x2 - offset, num_samples)))
    x3 = np.vstack((np.linspace(min_x1 + offset / multiple, max_x1, num_samples * multiple), max_x2 * np.ones(num_samples * multiple)))
    x4 = np.vstack((min_x1 * np.ones(num_samples), np.linspace(min_x2 + offset, max_x2, num_samples)))
    return np.hstack((x1, x2, x3, x4))


def generate_initial_conditions_dense_center(eq: np.ndarray, buffer: float, num_samples: int) -> np.ndarray:
    min_x1 = -buffer + eq[0]
    max_x1 = buffer + eq[0]
    min_x2 = -buffer + eq[1]
    max_x2 = buffer + eq[1]

    offset = buffer / num_samples
    x1_part1 = np.linspace(min_x1, -1.0, num_samples // 4, endpoint=False)
    x1_part2 = np.linspace(-1.0, 1.0, num_samples - 2 * (num_samples // 4), endpoint=False)
    x1_part3 = np.linspace(1.0, max_x1, num_samples // 4)
    x1_full = np.concatenate((x1_part1, x1_part2, x1_part3))

    x1 = np.vstack((x1_full, min_x2 * np.ones(num_samples)))
    x2 = np.vstack((max_x1 * np.ones(num_samples), np.linspace(min_x2, max_x2 - offset, num_samples)))
    x3 = np.vstack((x1_full[::-1], max_x2 * np.ones(num_samples)))
    x4 = np.vstack((min_x1 * np.ones(num_samples), np.linspace(min_x2 + offset, max_x2, num_samples)))
    return np.hstack((x1, x2, x3, x4))
