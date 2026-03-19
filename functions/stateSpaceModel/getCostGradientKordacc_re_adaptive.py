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


class CostGradientStructureCache:
    """
    Precomputes and caches sparse matrix structure to avoid rebuilding indices repeatedly.
    
    On each cost/gradient evaluation, only floating-point values are recomputed,
    not the sparsity pattern (row/col indices). This eliminates expensive list
    concatenation and sparse matrix assembly overhead.
    """
    
    def __init__(
        self,
        Traj_t: Sequence[Sequence[np.ndarray]],
        Ms_tot: int,
        n_cc: int,
        n_eig_total: int,
    ):
        """
        Args:
            Traj_t: Time vectors (cell structure).
            Ms_tot: Total number of measurement points.
            n_cc: Number of complex conjugate pairs.
            n_eig_total: Total number of eigenvalues (complex pairs + real).
        """
        self.Traj_t = Traj_t
        self.Ms_tot = Ms_tot
        self.n_cc = n_cc
        self.n_eig_total = n_eig_total
        self.Mt = sum(len(row) for row in Traj_t)
        self.numCols = n_eig_total * self.Mt
        
        # Precompute time blocks (reused across all evaluations)
        self.time_blocks = []
        self.block_shapes = []
        for jj, kk in _iter_traj_cells(Traj_t):
            t = np.asarray(Traj_t[jj][kk]).reshape(-1, 1)
            self.time_blocks.append(t)
            self.block_shapes.append(t.shape)
        
        # Precompute row/col index structure for L matrix
        self._precompute_L_indices()
    
    def _precompute_L_indices(self):
        """Precompute row/col indices for L matrix (constant across calls)."""
        rows_list = []
        cols_list = []
        
        i_prec = 0
        j_prec = 0
        
        # Complex conjugate pairs: 2 columns per pair
        for ii in range(0, 2 * self.n_cc, 2):
            for t in self.time_blocks:
                n_rows = t.shape[0]
                n_cols = 2  # [cos, -sin] for complex blocks
                
                row_idx = i_prec + np.tile(np.arange(n_rows), n_cols)
                col_idx = j_prec + np.repeat(np.arange(n_cols), n_rows)
                rows_list.append(row_idx)
                cols_list.append(col_idx)
                
                i_prec += n_rows
                j_prec += n_cols
        
        # Real eigenvalues: 1 column each
        n_real = self.n_eig_total - 2 * self.n_cc
        for ii in range(n_real):
            for t in self.time_blocks:
                n_rows = t.shape[0]
                n_cols = 1
                
                row_idx = i_prec + np.tile(np.arange(n_rows), n_cols)
                col_idx = j_prec + np.repeat(np.arange(n_cols), n_rows)
                rows_list.append(row_idx)
                cols_list.append(col_idx)
                
                i_prec += n_rows
                j_prec += n_cols
        
        # Precomputed index arrays (reused every evaluation)
        self.rows = np.concatenate(rows_list) if rows_list else np.array([], dtype=int)
        self.cols = np.concatenate(cols_list) if cols_list else np.array([], dtype=int)
    
    def compute_L_values(self, x: np.ndarray) -> np.ndarray:
        """
        Compute only the values for L matrix, reusing precomputed indices.
        
        Returns:
            values: Flattened values array for sparse matrix construction.
        """
        x = np.asarray(x, dtype=float).reshape(-1)
        vals_list = []
        
        # Complex conjugate pairs
        for ii in range(0, 2 * self.n_cc, 2):
            ReL = x[ii]
            ImL = x[ii + 1]
            
            for t in self.time_blocks:
                block = 2 * np.exp(ReL * t) * np.hstack((np.cos(ImL * t), -np.sin(ImL * t)))
                v = block.reshape(-1, order="F")
                vals_list.append(v)
        
        # Real eigenvalues
        for ii in range(2 * self.n_cc, len(x)):
            l = x[ii]
            
            for t in self.time_blocks:
                block = np.exp(l * t)
                v = block.reshape(-1, order="F")
                vals_list.append(v)
        
        return np.concatenate(vals_list) if vals_list else np.array([], dtype=complex)


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


def getCostGradientKordacc_re_adaptive_cached(
    x: np.ndarray,
    cache: CostGradientStructureCache,
    h: np.ndarray,
    lambda_reg: float = 1e-8,
) -> Tuple[float, np.ndarray]:
    """
    Compute cost and gradient using precomputed sparse L matrix structure.
    
    Reuses row/col indices to avoid expensive sparse matrix assembly.
    Only floating-point values are recomputed each call.
    
    Args:
        x: Eigenvalue parameters [Re1, Im1, Re2, Im2, ...].
        cache: CostGradientStructureCache with precomputed indices.
        h: Output vector.
        lambda_reg: Tikhonov regularization coefficient.
    
    Returns:
        (cost, gradient): Tuple of scalar cost and gradient array.
    """
    x = np.asarray(x, dtype=float).reshape(-1)
    h_vec = np.asarray(h, dtype=complex).reshape(-1, 1)
    
    # Build L using cached row/col indices, only recomputing values
    vals = cache.compute_L_values(x)
    L = sparse.coo_matrix((vals, (cache.rows, cache.cols)), shape=(cache.Ms_tot, cache.numCols))
    
    # Solve regularized system
    LL = L.T @ L + lambda_reg * sparse.eye(cache.numCols, format="csc")
    LL_inv = splinalg.inv(LL.tocsc())
    q = LL_inv @ (L.T @ h_vec)
    
    # Gradient computation (same logic as original)
    grad = np.zeros(len(x), dtype=complex)
    
    j_prec_R = 0
    j_prec_I = 0
    for ii in range(0, 2 * cache.n_cc, 2):
        ReL = x[ii]
        ImL = x[ii + 1]
        i_prec_R = 0
        i_prec_I = 0
        
        vals_R_list = []
        vals_Im_list = []
        rows_R_list = []
        cols_R_list = []
        rows_I_list = []
        cols_I_list = []
        
        for t in cache.time_blocks:
            dl_Re_block = 2 * t * np.exp(ReL * t) * np.hstack((np.cos(ImL * t), -np.sin(ImL * t)))
            dl_Im_block = 2 * t * np.exp(ReL * t) * np.hstack((-np.sin(ImL * t), -np.cos(ImL * t)))
            
            r_Re, c_Re, v_Re = _block_entries(dl_Re_block, i_prec_R, j_prec_R)
            r_Im, c_Im, v_Im = _block_entries(dl_Im_block, i_prec_I, j_prec_I)
            
            vals_R_list.append(v_Re)
            vals_Im_list.append(v_Im)
            rows_R_list.append(r_Re)
            cols_R_list.append(c_Re)
            rows_I_list.append(r_Im)
            cols_I_list.append(c_Im)
            
            i_prec_R += dl_Re_block.shape[0]
            j_prec_R += dl_Re_block.shape[1]
            i_prec_I += dl_Im_block.shape[0]
            j_prec_I += dl_Im_block.shape[1]
        
        dLdx_Re = sparse.coo_matrix(
            (np.concatenate(vals_R_list), (np.concatenate(rows_R_list), np.concatenate(cols_R_list))),
            shape=(cache.Ms_tot, cache.numCols),
        )
        dLdx_Im = sparse.coo_matrix(
            (np.concatenate(vals_Im_list), (np.concatenate(rows_I_list), np.concatenate(cols_I_list))),
            shape=(cache.Ms_tot, cache.numCols),
        )
        
        grad[ii] = h_vec.T @ dLdx_Re @ q + h_vec.T @ L @ (
            (-LL_inv @ (dLdx_Re.T @ L + L.T @ dLdx_Re) @ LL_inv) @ L.T @ h_vec + LL_inv @ dLdx_Re.T @ h_vec
        )
        
        grad[ii + 1] = h_vec.T @ dLdx_Im @ q + h_vec.T @ L @ (
            (-LL_inv @ (dLdx_Im.T @ L + L.T @ dLdx_Im) @ LL_inv) @ L.T @ h_vec + LL_inv @ dLdx_Im.T @ h_vec
        )
    
    j_prec = j_prec_R
    for ii in range(2 * cache.n_cc, len(x)):
        l = x[ii]
        i_prec = 0
        vals_d_list = []
        rows_d_list = []
        cols_d_list = []
        
        for t in cache.time_blocks:
            dl_real_block = t * np.exp(l * t)
            r, c, v = _block_entries(dl_real_block, i_prec, j_prec)
            vals_d_list.append(v)
            rows_d_list.append(r)
            cols_d_list.append(c)
            i_prec += dl_real_block.shape[0]
            j_prec += dl_real_block.shape[1]
        
        dLdx_real = sparse.coo_matrix(
            (np.concatenate(vals_d_list), (np.concatenate(rows_d_list), np.concatenate(cols_d_list))),
            shape=(cache.Ms_tot, cache.numCols),
        )
        
        grad[ii] = h_vec.T @ dLdx_real @ q + h_vec.T @ L @ (
            (-LL_inv @ (dLdx_real.T @ L + L.T @ dLdx_real) @ LL_inv) @ L.T @ h_vec + LL_inv @ dLdx_real.T @ h_vec
        )
    
    grad = -grad
    J = (h_vec.T @ h_vec - h_vec.T @ L @ q).item()
    
    return float(np.real_if_close(J)), np.real_if_close(grad).reshape(-1)

def getCostGradientKordacc_re_adaptive_cached(
    x: np.ndarray,
    cache: CostGradientStructureCache,
    h: np.ndarray,
    lambda_reg: float = 1e-8,
) -> Tuple[float, np.ndarray]:
    """
    Compute cost and gradient using precomputed sparse structure (fast path).
    
    This version reuses row/col indices across evaluations, avoiding expensive
    sparse matrix assembly. Only floating-point values are recomputed each call.
    
    Args:
        x: Eigenvalue parameters [Re1, Im1, Re2, Im2, ...].
        cache: CostGradientStructureCache object (created once per optimization).
        h: Output vector.
        lambda_reg: Regularization parameter.
    
    Returns:
        (cost, gradient) tuple.
    """
    x = np.asarray(x, dtype=float).reshape(-1)
    h_vec = np.asarray(h, dtype=complex).reshape(-1, 1)
    
    # Use cached indices, only recompute values
    vals, rows, cols, numCols = cache.compute_L_values_and_indices(x)
    L = sparse.coo_matrix((vals, (rows, cols)), shape=(cache.Ms_tot, numCols))
    
    # Solve with regularization (same as original)
    lambda_reg_val = lambda_reg
    LL = L.T @ L + lambda_reg_val * sparse.eye(numCols, format="csc")
    LL_inv = splinalg.inv(LL.tocsc())
    
    q = LL_inv @ (L.T @ h_vec)
    
    # Compute gradient (simplified path—reuses same structure idea)
    grad = np.zeros(len(x), dtype=complex)
    
    # Complex pair gradients
    for ii in range(0, 2 * cache.n_cc, 2):
        ReL = x[ii]
        ImL = x[ii + 1]
        
        # Build dL/dRe and dL/dIm efficiently
        vals_Re_list = []
        vals_Im_list = []
        rows_Re_list = []
        cols_Re_list = []
        rows_Im_list = []
        cols_Im_list = []
        
        i_prec = 0
        j_prec = ii // 2 * 2 * cache.Mt  # Column offset for this complex pair
        
        for t in cache.time_blocks:
            dl_Re_block = 2 * t * np.exp(ReL * t) * np.hstack((np.cos(ImL * t), -np.sin(ImL * t)))
            dl_Im_block = 2 * t * np.exp(ReL * t) * np.hstack((-np.sin(ImL * t), -np.cos(ImL * t)))
            
            v_Re = dl_Re_block.reshape(-1, order="F")
            v_Im = dl_Im_block.reshape(-1, order="F")
            vals_Re_list.append(v_Re)
            vals_Im_list.append(v_Im)
            
            n_rows, n_cols = dl_Re_block.shape
            row_idx = i_prec + np.tile(np.arange(n_rows), n_cols)
            col_idx = j_prec + np.repeat(np.arange(n_cols), n_rows)
            rows_Re_list.append(row_idx)
            cols_Re_list.append(col_idx)
            rows_Im_list.append(row_idx)
            cols_Im_list.append(col_idx)
            
            i_prec += n_rows
            j_prec += n_cols
        
        dLdx_Re = sparse.coo_matrix(
            (np.concatenate(vals_Re_list), (np.concatenate(rows_Re_list), np.concatenate(cols_Re_list))),
            shape=(cache.Ms_tot, numCols),
        )
        dLdx_Im = sparse.coo_matrix(
            (np.concatenate(vals_Im_list), (np.concatenate(rows_Im_list), np.concatenate(cols_Im_list))),
            shape=(cache.Ms_tot, numCols),
        )
        
        grad[ii] = h_vec.T @ dLdx_Re @ q + h_vec.T @ L @ (
            (-LL_inv @ (dLdx_Re.T @ L + L.T @ dLdx_Re) @ LL_inv) @ L.T @ h_vec + LL_inv @ dLdx_Re.T @ h_vec
        )
        
        grad[ii + 1] = h_vec.T @ dLdx_Im @ q + h_vec.T @ L @ (
            (-LL_inv @ (dLdx_Im.T @ L + L.T @ dLdx_Im) @ LL_inv) @ L.T @ h_vec + LL_inv @ dLdx_Im.T @ h_vec
        )
    
    # Real eigenvalue gradients
    j_prec = 2 * cache.n_cc * cache.Mt
    for ii in range(2 * cache.n_cc, len(x)):
        l = x[ii]
        
        vals_d_list = []
        rows_d_list = []
        cols_d_list = []
        
        i_prec = 0
        for t in cache.time_blocks:
            dl_real_block = t * np.exp(l * t)
            v_d = dl_real_block.reshape(-1, order="F")
            vals_d_list.append(v_d)
            
            n_rows, n_cols = dl_real_block.shape
            row_idx = i_prec + np.tile(np.arange(n_rows), n_cols)
            col_idx = j_prec + np.repeat(np.arange(n_cols), n_rows)
            rows_d_list.append(row_idx)
            cols_d_list.append(col_idx)
            
            i_prec += n_rows
            j_prec += n_cols
        
        dLdx_real = sparse.coo_matrix(
            (np.concatenate(vals_d_list), (np.concatenate(rows_d_list), np.concatenate(cols_d_list))),
            shape=(cache.Ms_tot, numCols),
        )
        
        grad[ii] = h_vec.T @ dLdx_real @ q + h_vec.T @ L @ (
            (-LL_inv @ (dLdx_real.T @ L + L.T @ dLdx_real) @ LL_inv) @ L.T @ h_vec + LL_inv @ dLdx_real.T @ h_vec
        )
    
    grad = -grad
    J = (h_vec.T @ h_vec - h_vec.T @ L @ q).item()
    
    return float(np.real_if_close(J)), np.real_if_close(grad).reshape(-1)