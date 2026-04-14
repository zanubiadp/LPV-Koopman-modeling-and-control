"""
Compute LPV input matrix via 3D numerical differentiation of eigenfunctions.

Python translation of MATLAB's `build_B_numgrad_3D`.
Only 4th-order central differences are implemented.
"""
from __future__ import annotations

import numpy as np
from collections.abc import Callable, Sequence
from scipy.interpolate import RegularGridInterpolator


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

    # Batch evaluation: evaluate all eigenfunctions at each stencil point
    # instead of evaluating one eigenfunction at multiple points.
    # This reduces function call overhead and improves cache locality.
    
    stencil_z = np.array([
        [x_val, y_val, z_val + 2 * h],
        [x_val, y_val, z_val + h],
        [x_val, y_val, z_val - h],
        [x_val, y_val, z_val - 2 * h],
    ])
    
    stencil_x = np.array([
        [x_val + 2 * h, y_val, z_val],
        [x_val + h, y_val, z_val],
        [x_val - h, y_val, z_val],
        [x_val - 2 * h, y_val, z_val],
    ])
    
    stencil_y = np.array([
        [x_val, y_val + 2 * h, z_val],
        [x_val, y_val + h, z_val],
        [x_val, y_val - h, z_val],
        [x_val, y_val - 2 * h, z_val],
    ])
    
    # Evaluate all eigenfunctions at all z stencil points at once
    phi_z = np.array([np.array([f(pt) for f in F]) for pt in stencil_z])  # (4, total_eigs)
    
    # Compute dF/dz for all eigenfunctions in one go
    df_dz_all = (
        -phi_z[0, :] + 8 * phi_z[1, :] - 8 * phi_z[2, :] + phi_z[3, :]
    ) / (12 * h)
    Jz[:, 0] = df_dz_all
    
    if gc[0] != 0 and gc[1] != 0:
        # Evaluate at x stencil points
        phi_x = np.array([np.array([f(pt) for f in F]) for pt in stencil_x])
        
        # Evaluate at y stencil points
        phi_y = np.array([np.array([f(pt) for f in F]) for pt in stencil_y])
        
        # Compute dF/dx and dF/dy for all eigenfunctions
        df_dx_all = (
            -phi_x[0, :] + 8 * phi_x[1, :] - 8 * phi_x[2, :] + phi_x[3, :]
        ) / (12 * h)
        
        df_dy_all = (
            -phi_y[0, :] + 8 * phi_y[1, :] - 8 * phi_y[2, :] + phi_y[3, :]
        ) / (12 * h)
        
        J[:, 0] = df_dx_all
        J[:, 1] = df_dy_all
        J[:, 2] = df_dz_all

    if gc[0] != 0 and gc[1] != 0:
        return J @ gc

    return Jz


def precompute_B_grid(
    h: float,
    F: Sequence[Callable[[np.ndarray], np.ndarray]],
    nEig: Sequence[int],
    gc: np.ndarray,
    x1_range: tuple[float, float],
    x2_range: tuple[float, float],
    u_range: tuple[float, float],
    x1_pts: int = 25,
    x2_pts: int = 25,
    u_pts: int = 25,
    verbose: bool = True,
) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray], np.ndarray]:
    """
    Pre-compute B matrices on a regular (x1, x2, u) grid for fast interpolation.
    
    This function evaluates B at grid points once, enabling O(1) lookups during
    prediction via RegularGridInterpolator (instead of computing B each time).
    
    Args:
        h: Step size for finite differences.
        F: Sequence of eigenfunction callables.
        nEig: Eigenvalue counts per state.
        gc: Input affine coefficient vector.
        x1_range: (min, max) for x1 dimension.
        x2_range: (min, max) for x2 dimension.
        u_range: (min, max) for u dimension.
        x1_pts: Number of grid points for x1.
        x2_pts: Number of grid points for x2.
        u_pts: Number of grid points for u.
        verbose: Print progress.
    
    Returns:
        (grid_axes, B_grid) where:
            grid_axes: Tuple of (x1_grid, x2_grid, u_grid) arrays
            B_grid: Array of shape (x1_pts, x2_pts, u_pts, n_eig, 1) with B matrices
    """
    total_eigs = int(np.sum(nEig))
    
    # Create grid axes
    x1_grid = np.linspace(x1_range[0], x1_range[1], x1_pts)
    x2_grid = np.linspace(x2_range[0], x2_range[1], x2_pts)
    u_grid = np.linspace(u_range[0], u_range[1], u_pts)
    
    # Allocate result array: (x1, x2, u, n_eig, 1)
    B_grid = np.zeros((x1_pts, x2_pts, u_pts, total_eigs, 1), dtype=complex)
    
    total_pts = x1_pts * x2_pts * u_pts
    computed = 0
    
    # Compute B at each grid point
    for i, x1 in enumerate(x1_grid):
        for j, x2 in enumerate(x2_grid):
            for k, u in enumerate(u_grid):
                x_state = np.array([x1, x2, u])
                B_mat = build_B_numgrad_3D(h, F, x_state, nEig, gc)
                
                # Handle both full matrix and vector returns
                if isinstance(B_mat, np.ndarray):
                    if B_mat.shape[1] == 1:
                        B_grid[i, j, k, :, 0] = B_mat.ravel()
                    else:
                        B_grid[i, j, k, :, 0] = B_mat @ gc
                
                computed += 1
                if verbose and computed % max(1, total_pts // 10) == 0:
                    print(f"  Precomputed B grid: {computed}/{total_pts} points")
    
    if verbose:
        print(f"✓ B matrix grid pre-computation complete ({total_pts} points)")
    
    return (x1_grid, x2_grid, u_grid), B_grid


class BMatrixInterpolator:
    """
    Fast interpolator for pre-computed B matrices on a regular grid.
    
    Instead of calling build_B_numgrad_3D for each prediction step
    (expensive: 12 eigenfunction evaluations), this uses grid interpolation
    (O(1) lookup after precomputation).
    """
    
    def __init__(
        self,
        grid_axes: tuple[np.ndarray, np.ndarray, np.ndarray],
        B_grid: np.ndarray,
    ):
        """
        Args:
            grid_axes: Tuple of (x1_grid, x2_grid, u_grid).
            B_grid: Array of shape (x1_pts, x2_pts, u_pts, n_eig, 1).
        """
        self.x1_grid, self.x2_grid, self.u_grid = grid_axes
        self.B_grid = B_grid
        self.n_eig = B_grid.shape[3]
        
        # Build interpolators for each eigenfunction component
        self.interpolators = []
        for eig_idx in range(self.n_eig):
            interp = RegularGridInterpolator(
                points=(self.x1_grid, self.x2_grid, self.u_grid),
                values=B_grid[:, :, :, eig_idx, 0],
                bounds_error=True,
                method="linear",
            )
            self.interpolators.append(interp)
    
    def __call__(self, x_state: np.ndarray) -> np.ndarray:
        """
        Evaluate B matrix at arbitrary state point via grid interpolation.
        Clips out-of-bounds queries to grid boundaries to avoid NaN values.
        
        Args:
            x_state: State vector [x1, x2, u].
        
        Returns:
            B matrix (n_eig, 1) at the given state.
        """
        x_state = np.asarray(x_state, dtype=float).ravel()
        x1, x2, u = x_state[:3]
        
        # Clip to grid bounds to avoid out-of-bounds issues
        x1 = np.clip(x1, self.x1_grid[0], self.x1_grid[-1])
        x2 = np.clip(x2, self.x2_grid[0], self.x2_grid[-1])
        u = np.clip(u, self.u_grid[0], self.u_grid[-1])
        
        query_point = np.array([[x1, x2, u]])
        
        # Evaluate all eigenfunction components
        B_result = np.zeros((self.n_eig, 1), dtype=complex)
        for eig_idx, interp in enumerate(self.interpolators):
            B_result[eig_idx, 0] = interp(query_point)[0]
        
        return B_result
