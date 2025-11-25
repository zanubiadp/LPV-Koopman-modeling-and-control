"""
Compute the observability matrix for a discrete-time linear system.
"""
from __future__ import annotations

import numpy as np


def observability(A: np.ndarray, C: np.ndarray, k: int) -> np.ndarray:
    """
    Build the observability matrix:
        O = [C; C A; C A^2; ...; C A^(k-1)]

    Args:
        A: State matrix (n x n).
        C: Output matrix (m x n).
        k: Number of block rows to include.

    Returns:
        Observability matrix with shape (m * k, n).
    """
    nc = C.shape[0]
    na = A.shape[0]
    O = np.zeros((nc * k, na), dtype=float)

    for ii in range(1, k + 1):
        idx_start = (ii - 1) * nc
        idx_end = idx_start + nc
        O[idx_start:idx_end, :] = C @ np.linalg.matrix_power(A, ii - 1)

    return O
