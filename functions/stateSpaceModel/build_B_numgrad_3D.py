"""
Compute LPV input matrix via 3D numerical differentiation of eigenfunctions.

Python translation of MATLAB's `build_B_numgrad_3D`.
Only 4th-order central differences are implemented.
"""
from __future__ import annotations

import numpy as np
from collections.abc import Callable, Sequence


def build_B_numgrad_3D(
    h: float,
    F: Sequence[Callable[[np.ndarray], np.ndarray]],
    x_initial: np.ndarray,
    nEig: Sequence[int],
    gc: np.ndarray,
) -> np.ndarray:
    """
    Numerically derive eigenfunctions to build the LPV input matrix (3D case).

    Args:
        h: Step size for finite differences.
        F: Sequence of eigenfunction callables, each taking a 3-vector.
        x_initial: Starting point [x, y, z].
        nEig: Iterable with counts of eigenvalues per state (sum = len(F)).
        gc: Input affine coefficient matrix.

    Returns:
        B_numgrad (matrix) or Jz (vector) depending on gc.
    """
    x_val, y_val, z_val = map(float, x_initial[:3])
    total_eigs = int(np.sum(nEig))
    # Use complex dtype so any imaginary parts returned by eigenfunction
    # interpolants are preserved during finite-difference calculations.
    J = np.zeros((total_eigs, 3), dtype=complex)
    Jz = np.zeros((total_eigs, 1), dtype=complex)

    for ii in range(total_eigs):
        f = F[ii]

        # Partial derivative w.r.t. z (4th-order central difference)
        df_dz = (
            -f(np.array([x_val, y_val, z_val + 2 * h]))
            + 8 * f(np.array([x_val, y_val, z_val + h]))
            - 8 * f(np.array([x_val, y_val, z_val - h]))
            + f(np.array([x_val, y_val, z_val - 2 * h]))
        ) / (12 * h)
        Jz[ii, 0] = df_dz

        if gc[0] != 0 and gc[1] != 0:
            df_dx = (
                -f(np.array([x_val + 2 * h, y_val, z_val]))
                + 8 * f(np.array([x_val + h, y_val, z_val]))
                - 8 * f(np.array([x_val - h, y_val, z_val]))
                + f(np.array([x_val - 2 * h, y_val, z_val]))
            ) / (12 * h)

            df_dy = (
                -f(np.array([x_val, y_val + 2 * h, z_val]))
                + 8 * f(np.array([x_val, y_val + h, z_val]))
                - 8 * f(np.array([x_val, y_val - h, z_val]))
                + f(np.array([x_val, y_val - 2 * h, z_val]))
            ) / (12 * h)

            J[ii, :] = [df_dx, df_dy, df_dz]

    if gc[0] != 0 and gc[1] != 0:
        return J @ gc

    return Jz
