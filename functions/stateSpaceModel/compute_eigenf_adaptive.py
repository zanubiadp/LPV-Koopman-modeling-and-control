"""
Python translation of MATLAB's `compute_eigenf_adaptive`.
Computes eigenfunction values and interpolants from learned eigenvalues.
"""
from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.interpolate import LinearNDInterpolator
from scipy.sparse import linalg as splinalg
from typing import Callable, Iterable, Sequence, Tuple, List

from functions.general.building_L_Lambda_adaptive import building_L_Lambda_adaptive


def compute_eigenf_adaptive(
    lambdas: Sequence[Sequence[complex]],
    Traj_t: Sequence[Sequence[np.ndarray]],
    data: np.ndarray,
    interp_idx: Iterable[int] | None = None,
    h_idx: Iterable[int] | None = None,
) -> Tuple[np.ndarray, List[Callable[[np.ndarray], np.ndarray]]]:
    """
    Args:
        lambdas: Iterable of eigenvalue groups (one group per state).
        Traj_t: Cell-like structure of time vectors for each trajectory.
        data: State/measurement data matrix (states x total_samples).
        interp_idx: Indices of dimensions used for interpolation (default all rows).
        h_idx: Row indices in `data` corresponding to each eigenvalue group.

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

    g_list = []
    for L_mat, h_vec in zip(L_Lambda, h_list):
        normal_mat = L_mat.T @ L_mat
        rhs = L_mat.T @ h_vec
        try:
            g_vec = splinalg.spsolve(normal_mat.tocsc(), rhs)
        except Exception:
            # Fallback to least squares if the system is singular
            g_vec = splinalg.lsqr(normal_mat, rhs)[0]
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

    # Build interpolants
    phi_hat: List[Callable[[np.ndarray], np.ndarray]] = []
    if len(interp_idx) > 3:
        raise NotImplementedError("Case length(interp_idx) > 3 not implemented yet.")

    points = np.asarray(data[interp_idx, :].T, dtype=float)
    for ii in range(phi_vals.shape[0]):
        values = phi_vals[ii, :]
        interpolator = LinearNDInterpolator(points, values)

        def make_fn(interp):
            def fn(x: np.ndarray) -> np.ndarray:
                x_arr = np.asarray(x, dtype=float)
                vals = interp(x_arr.T)
                return np.asarray(vals).reshape(-1)

            return fn

        phi_hat.append(make_fn(interpolator))

    return phi_vals, phi_hat
