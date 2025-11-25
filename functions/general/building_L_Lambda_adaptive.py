"""
Construct the sparse L_Lambda matrix (adaptive) translated from MATLAB.
"""
from __future__ import annotations

import numpy as np
from scipy import sparse
from typing import Sequence


def building_L_Lambda_adaptive(
    lambdas: Sequence[complex],
    Mt: int,
    Ms_vec: Sequence[int],
    Traj_len_tot: int,
):
    """
    Python translation of the MATLAB `building_L_Lambda_adaptive` function.

    Args:
        lambdas: Iterable of eigenvalues (can be complex).
        Mt: Number of trajectories.
        Ms_vec: Iterable of sample counts (per trajectory) minus 1.
        Traj_len_tot: Total trajectory length (sum of Ms_vec + 1).

    Returns:
        scipy.sparse.coo_matrix representing L_Lambda.
    """
    lambdas = np.asarray(lambdas, dtype=complex)
    Ms_vec = np.asarray(Ms_vec, dtype=int)

    N = lambdas.size

    max_rows = Traj_len_tot
    max_cols = N * Mt

    nnz = Traj_len_tot * N
    i_idx = np.zeros(nnz, dtype=int)
    j_idx = np.zeros(nnz, dtype=int)
    values = np.zeros(nnz, dtype=complex)

    for ii in range(1, N + 1):
        idx_start_i = (ii - 1) * Traj_len_tot
        idx_end_i = ii * Traj_len_tot
        i_idx[idx_start_i:idx_end_i] = np.arange(Traj_len_tot)

        idx_start_j = idx_start_i
        for jj, Ms in enumerate(Ms_vec, start=1):
            idx_end_j = idx_start_j + Ms
            j_idx[idx_start_j:idx_end_j + 1] = ( (ii - 1) * Mt + jj - 1 )
            values[idx_start_j:idx_end_j + 1] = lambdas[ii - 1] ** np.arange(Ms + 1)
            idx_start_j = idx_end_j + 1

    L_Lambda = sparse.coo_matrix((values, (i_idx, j_idx)), shape=(max_rows, max_cols))
    return L_Lambda
