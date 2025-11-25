"""
Compute LPV input matrix via numerical differentiation of eigenfunctions.

Python translation of MATLAB's `build_B_numgrad`.
"""
from __future__ import annotations

import numpy as np
from collections.abc import Callable, Sequence


def build_B_numgrad(
    order: int,
    h: float,
    F: Sequence[Callable[[np.ndarray], np.ndarray]],
    x_initial: np.ndarray,
    nEig: Sequence[int],
    gc: np.ndarray,
) -> np.ndarray:
    """
    Numerically derive eigenfunctions to build the LPV input matrix.

    Args:
        order: Finite difference order (1, 2, or 4).
        h: Step size for finite differences.
        F: Sequence of eigenfunction callables, each taking a column vector.
        x_initial: Starting point [x, y] (array-like).
        nEig: Iterable with counts of eigenvalues per state (sum = len(F)).
        gc: Input affine coefficient matrix (2 x m).

    Returns:
        B_numgrad matrix (sum(nEig) x gc.shape[1]).
    """
    x_val = float(x_initial[0])
    y_val = float(x_initial[1])
    total_eigs = int(np.sum(nEig))
    J = np.zeros((total_eigs, 2), dtype=float)

    for ii in range(total_eigs):
        f = F[ii]

        if order == 1:
            z_val = f(np.array([x_val, y_val]))
            dz_dx = (f(np.array([x_val + h, y_val])) - z_val) / h
            dz_dy = (f(np.array([x_val, y_val + h])) - z_val) / h
        elif order == 2:
            dz_dx = (f(np.array([x_val + h, y_val])) - f(np.array([x_val - h, y_val]))) / (2 * h)
            dz_dy = (f(np.array([x_val, y_val + h])) - f(np.array([x_val, y_val - h]))) / (2 * h)
        elif order == 4:
            dz_dx = (
                -f(np.array([x_val + 2 * h, y_val]))
                + 8 * f(np.array([x_val + h, y_val]))
                - 8 * f(np.array([x_val - h, y_val]))
                + f(np.array([x_val - 2 * h, y_val]))
            ) / (12 * h)
            dz_dy = (
                -f(np.array([x_val, y_val + 2 * h]))
                + 8 * f(np.array([x_val, y_val + h]))
                - 8 * f(np.array([x_val, y_val - h]))
                + f(np.array([x_val, y_val - 2 * h]))
            ) / (12 * h)
        else:
            raise ValueError("order must be one of {1, 2, 4}")

        J[ii, :] = [dz_dx, dz_dy]

    B_numgrad = J @ gc
    return B_numgrad
