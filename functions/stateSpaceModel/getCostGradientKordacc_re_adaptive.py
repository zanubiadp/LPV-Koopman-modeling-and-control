"""
Python translation of MATLAB's `getCostGradientKordacc_re_adaptive`.

Computes cost, gradient, and sparse matrix L for adaptive eigenvalue
learning from trajectory data.
"""
from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.sparse import linalg as splinalg
from typing import Iterable, Sequence, Tuple


def _block_entries(mat: np.ndarray, row_offset: int, col_offset: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return row/col/value arrays (0-based) for a dense block."""
    rows, cols = mat.shape
    row_idx = row_offset + np.tile(np.arange(rows), cols)
    col_idx = col_offset + np.repeat(np.arange(cols), rows)
    values = mat.reshape(-1, order="F")  # column-major flatten
    return row_idx, col_idx, values


def _iter_traj_cells(container: Sequence[Sequence]) -> Iterable[Tuple[int, int]]:
    for jj in range(len(container)):
        for kk in range(len(container[jj])):
            yield jj, kk


def getCostGradientKordacc_re_adaptive(
    x: np.ndarray,
    Traj: Sequence[Sequence[np.ndarray]],
    Traj_t: Sequence[Sequence[np.ndarray]],
    h: np.ndarray,
    n_cc: int,
) -> Tuple[float, np.ndarray, sparse.coo_matrix]:
    """
    Args:
        x: Vector of eigenvalue parameters [Re1, Im1, Re2, Im2, Re3, Re4, ...].
        Traj: Trajectories (cell array equivalent) supporting varying lengths.
        Traj_t: Time vectors corresponding to each trajectory.
        h: Output vector stacked across trajectories (length Ms_tot).
        n_cc: Number of complex conjugate pairs (>= 0).

    Returns:
        J: Cost scalar.
        grad: Gradient vector (same length as x).
        L: Sparse matrix used in the cost.
    """
    x = np.asarray(x, dtype=float).reshape(-1)
    h_vec = np.asarray(h, dtype=complex).reshape(-1, 1)

    Mt = sum(len(row) for row in Traj_t)  # total number of trajectories
    Ms_tot = h_vec.shape[0]

    if 2 * n_cc > len(x):
        raise ValueError("number of n_cc couples > length(x)/2")

    numCols = len(x) * Mt

    # Build L matrix
    rows_list: list[np.ndarray] = []
    cols_list: list[np.ndarray] = []
    vals_list: list[np.ndarray] = []

    j_prec = 0
    for ii in range(0, 2 * n_cc, 2):
        ReL = x[ii]
        ImL = x[ii + 1]
        i_prec = 0

        for jj, kk in _iter_traj_cells(Traj_t):
            t = np.asarray(Traj_t[jj][kk]).reshape(-1, 1)
            block = 2 * np.exp(ReL * t) * np.hstack((np.cos(ImL * t), -np.sin(ImL * t)))
            r, c, v = _block_entries(block, i_prec, j_prec)
            rows_list.append(r)
            cols_list.append(c)
            vals_list.append(v)
            i_prec += block.shape[0]
            j_prec += block.shape[1]

    for ii in range(2 * n_cc, len(x)):
        l = x[ii]
        i_prec = 0

        for jj, kk in _iter_traj_cells(Traj_t):
            t = np.asarray(Traj_t[jj][kk]).reshape(-1, 1)
            block = np.exp(l * t)
            r, c, v = _block_entries(block, i_prec, j_prec)
            rows_list.append(r)
            cols_list.append(c)
            vals_list.append(v)
            i_prec += block.shape[0]
            j_prec += block.shape[1]

    rows = np.concatenate(rows_list) if rows_list else np.array([], dtype=int)
    cols = np.concatenate(cols_list) if cols_list else np.array([], dtype=int)
    vals = np.concatenate(vals_list) if vals_list else np.array([], dtype=complex)

    L = sparse.coo_matrix((vals, (rows, cols)), shape=(Ms_tot, numCols))

    # Small Tikhonov regularization to avoid exact singularity in LL = L.T @ L
    # (prevents SuperLU 'Factor is exactly singular' when LL is rank-deficient).
    lambda_reg = 1e-8
    LL = L.T @ L + lambda_reg * sparse.eye(numCols, format="csc")
    # Solve for the inverse implicitly
    LL_inv = splinalg.inv(LL.tocsc())

    q = LL_inv @ (L.T @ h_vec)

    grad = np.zeros(len(x), dtype=complex)

    # Gradients for complex variables
    j_prec_R = 0
    j_prec_I = 0
    for ii in range(0, 2 * n_cc, 2):
        ReL = x[ii]
        ImL = x[ii + 1]
        i_prec_R = 0
        i_prec_I = 0

        rows_R_list: list[np.ndarray] = []
        cols_R_list: list[np.ndarray] = []
        vals_R_list: list[np.ndarray] = []

        rows_I_list: list[np.ndarray] = []
        cols_I_list: list[np.ndarray] = []
        vals_I_list: list[np.ndarray] = []

        for jj, kk in _iter_traj_cells(Traj_t):
            t = np.asarray(Traj_t[jj][kk]).reshape(-1, 1)
            dl_Re_block = 2 * t * np.exp(ReL * t) * np.hstack((np.cos(ImL * t), -np.sin(ImL * t)))
            dl_Im_block = 2 * t * np.exp(ReL * t) * np.hstack((-np.sin(ImL * t), -np.cos(ImL * t)))

            r, c, v = _block_entries(dl_Re_block, i_prec_R, j_prec_R)
            rows_R_list.append(r)
            cols_R_list.append(c)
            vals_R_list.append(v)

            r, c, v = _block_entries(dl_Im_block, i_prec_I, j_prec_I)
            rows_I_list.append(r)
            cols_I_list.append(c)
            vals_I_list.append(v)

            i_prec_R += dl_Re_block.shape[0]
            j_prec_R += dl_Re_block.shape[1]
            i_prec_I += dl_Im_block.shape[0]
            j_prec_I += dl_Im_block.shape[1]

        dLdx_Re = sparse.coo_matrix(
            (np.concatenate(vals_R_list), (np.concatenate(rows_R_list), np.concatenate(cols_R_list))),
            shape=(Ms_tot, numCols),
        )
        dLdx_Im = sparse.coo_matrix(
            (np.concatenate(vals_I_list), (np.concatenate(rows_I_list), np.concatenate(cols_I_list))),
            shape=(Ms_tot, numCols),
        )

        grad[ii] = h_vec.T @ dLdx_Re @ q + h_vec.T @ L @ (
            (-LL_inv @ (dLdx_Re.T @ L + L.T @ dLdx_Re) @ LL_inv) @ L.T @ h_vec + LL_inv @ dLdx_Re.T @ h_vec
        )

        grad[ii + 1] = h_vec.T @ dLdx_Im @ q + h_vec.T @ L @ (
            (-LL_inv @ (dLdx_Im.T @ L + L.T @ dLdx_Im) @ LL_inv) @ L.T @ h_vec + LL_inv @ dLdx_Im.T @ h_vec
        )

    j_prec = j_prec_R
    # Gradients for real variables
    for ii in range(2 * n_cc, len(x)):
        l = x[ii]
        i_prec = 0
        rows_d_list: list[np.ndarray] = []
        cols_d_list: list[np.ndarray] = []
        vals_d_list: list[np.ndarray] = []

        for jj, kk in _iter_traj_cells(Traj_t):
            t = np.asarray(Traj_t[jj][kk]).reshape(-1, 1)
            dl_real_block = t * np.exp(l * t)

            r, c, v = _block_entries(dl_real_block, i_prec, j_prec)
            rows_d_list.append(r)
            cols_d_list.append(c)
            vals_d_list.append(v)

            i_prec += dl_real_block.shape[0]
            j_prec += dl_real_block.shape[1]

        dLdx_real = sparse.coo_matrix(
            (np.concatenate(vals_d_list), (np.concatenate(rows_d_list), np.concatenate(cols_d_list))),
            shape=(Ms_tot, numCols),
        )

        grad[ii] = h_vec.T @ dLdx_real @ q + h_vec.T @ L @ (
            (-LL_inv @ (dLdx_real.T @ L + L.T @ dLdx_real) @ LL_inv) @ L.T @ h_vec + LL_inv @ dLdx_real.T @ h_vec
        )

    grad = -grad

    J = (h_vec.T @ h_vec - h_vec.T @ L @ q).item()

    return float(np.real_if_close(J)), np.real_if_close(grad).reshape(-1), L
