"""
Python translation of MATLAB's `compute_eigenf_adaptive`.
Computes eigenfunction values and interpolants from learned eigenvalues.
"""
from __future__ import annotations

import numpy as np
from functools import lru_cache

from scipy import sparse
from scipy.interpolate import LinearNDInterpolator, RegularGridInterpolator
from scipy.sparse import linalg as splinalg
from typing import Callable, Iterable, Sequence, Tuple, List

from functions.general.building_L_Lambda_adaptive import building_L_Lambda_adaptive


def compute_eigenf_adaptive(
    lambdas: Sequence[Sequence[complex]],
    Traj_t: Sequence[Sequence[np.ndarray]],
    data: np.ndarray,
    interp_idx: Iterable[int] | None = None,
    h_idx: Iterable[int] | None = None,
    lambda_reg = 1e-15,  # regularization parameter, tune as needed
    interp_cache_size: int = 16_384,  # number of cached interpolation points per eigenfunction
) -> Tuple[np.ndarray, List[Callable[[np.ndarray], np.ndarray]]]:
    """
    Args:
        lambdas: Iterable of eigenvalue groups (one group per state).
        Traj_t: Cell-like structure of time vectors for each trajectory.
        data: State/measurement data matrix (states x total_samples).
        interp_idx: Indices of dimensions used for interpolation (default all rows).
        h_idx: Row indices in `data` corresponding to each eigenvalue group.
        interp_cache_size: Maximum number of cached interpolation points per eigenfunction.

    Returns:
        phi_vals: Array of eigenfunction values (sum(nEig) x Traj_len_tot).
        phi_hat: List of interpolant callables, one per eigenfunction.
    """
    if interp_idx is None:
        interp_idx = range(data.shape[0])
    if h_idx is None:
        h_idx = range(len(lambdas))

    interp_idx = list(interp_idx)
    h_idx = list(h_idx)

    if len(h_idx) != len(lambdas):
        raise ValueError("Careful with indices, go check them.")

    nEig = [len(group) for group in lambdas]

    # Total number of trajectories (flattened cell count)
    Mt = sum(len(row) for row in Traj_t)

    # Build Ms_vec in the same ordering as MATLAB loops
    Ms_vec = []
    for ii in range(len(Traj_t)):
        for jj in range(len(Traj_t[ii])):
            Ms_vec.append(len(Traj_t[ii][jj]) - 1)
    Ms_vec = np.asarray(Ms_vec, dtype=int)
    Traj_len_tot = int(np.sum(Ms_vec) + Mt)

    # Boundary functions computation
    L_Lambda = []
    h_list = []
    for idx, lam_group in enumerate(lambdas):
        L_Lambda.append(building_L_Lambda_adaptive(lam_group, Mt, Ms_vec, Traj_len_tot))
        h_list.append(np.asarray(data[h_idx[idx], :], dtype=complex).reshape(-1, 1))

    # Solve with regularization to avoid singular matrix inversion
    g_list = []

    for L_mat, h_vec in zip(L_Lambda, h_list):
        normal_mat = L_mat.T @ L_mat
        rhs = L_mat.T @ h_vec
        
        # Add regularization to the normal equations
        normal_mat_reg = normal_mat + lambda_reg * sparse.eye(normal_mat.shape[0])
        
        try:
            g_vec = splinalg.spsolve(normal_mat_reg.tocsc(), rhs)
        except Exception:
            g_vec = splinalg.lsqr(normal_mat_reg, rhs)[0]
        
        g_list.append(np.asarray(g_vec).reshape(-1, 1))

    # Flatten all eigenvalues
    Lambda = np.concatenate([np.asarray(group, dtype=complex) for group in lambdas]).reshape(-1)

    phi_vals = np.zeros((np.sum(nEig), Traj_len_tot), dtype=complex)

    eig_prec = 0  # cumulative eigenvalue offset
    for ii, n_eig_state in enumerate(nEig):
        for eig_s in range(n_eig_state):
            eig_t = eig_s + eig_prec
            istart = Mt * eig_s
            phistart = 0
            for jj, Ms in enumerate(Ms_vec):
                phiend = phistart + Ms + 1
                coeff_idx = istart + jj  # zero-based
                coeff = g_list[ii][coeff_idx, 0]
                powers = np.power(Lambda[eig_t], np.arange(Ms + 1))
                phi_vals[eig_t, phistart:phiend] = powers * coeff
                phistart = phiend
        eig_prec += n_eig_state

    # After computing phi_vals, create a regular grid
    x1_grid = np.linspace(data[interp_idx[0], :].min(), data[interp_idx[0], :].max(), 20)
    x2_grid = np.linspace(data[interp_idx[1], :].min(), data[interp_idx[1], :].max(), 20)
    # (adjust grid density as needed)

    X1, X2 = np.meshgrid(x1_grid, x2_grid, indexing='ij')
    grid_points = np.stack([X1.ravel(), X2.ravel()], axis=1)

    # Evaluate eigenfunctions on grid using existing LinearNDInterpolator
    phi_grid = []
    for old_interp in phi_hat:  # The slow interpolators
        grid_vals = old_interp(grid_points).reshape(X1.shape)
        phi_grid.append(grid_vals)

    # Now replace with fast grid interpolators
    phi_hat = []
    for grid_vals in phi_grid:
        fast_interp = RegularGridInterpolator((x1_grid, x2_grid), grid_vals, 
                                            bounds_error=False, fill_value='extrapolate')
        phi_hat.append(fast_interp)

    return phi_vals, phi_hat
