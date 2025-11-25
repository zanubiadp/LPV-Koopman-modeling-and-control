"""
Compute MPC cost matrices for LPV systems (translated from MATLAB).
"""
from __future__ import annotations

import numpy as np


def linear_cost_matrices_K_lpv(
    A: np.ndarray,
    B: np.ndarray,
    C: np.ndarray,
    C_r: np.ndarray,
    N: int,
    P: np.ndarray,
    Q: np.ndarray,
    R: np.ndarray,
    z_k: np.ndarray,
    r_N: np.ndarray,
    INPUT_WEIGHT: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Translation of MATLAB's `linear_cost_matrices_K_lpv`.

    Args:
        A, B, C, C_r: System matrices.
        N: Horizon length.
        P, Q, R: Cost matrices.
        z_k: Current augmented state (column vector).
        r_N: Reference trajectory stacked over the horizon.
        INPUT_WEIGHT: Weighting scheme ('delta' or 'deltadot').

    Returns:
        H, f, Phi, Gamma matrices for the quadratic cost.
    """
    A = np.asarray(A, dtype=complex)
    B = np.asarray(B, dtype=complex)
    C = np.asarray(C, dtype=complex)
    C_r = np.asarray(C_r, dtype=complex)
    P = np.asarray(P, dtype=complex)
    Q = np.asarray(Q, dtype=complex)
    R = np.asarray(R, dtype=complex)
    z_k = np.asarray(z_k, dtype=complex).reshape(-1, 1)
    r_N = np.asarray(r_N, dtype=complex).reshape(-1, 1)

    n = A.shape[0]
    m = B.shape[1]
    q = 1

    C_d = np.array([[0, 0, 1]], dtype=complex)

    C_big_r = np.kron(np.eye(N, dtype=complex), C_r @ C)
    C_big_d = np.kron(np.eye(N, dtype=complex), C_d @ C)

    Phi = np.zeros((n * N, n), dtype=complex)
    Gamma = np.zeros((n * N, N * m), dtype=complex)
    for i in range(1, N + 1):
        Phi[(i - 1) * n : i * n, :] = np.linalg.matrix_power(A, i)

    for i in range(1, N + 1):
        for j in range(1, i + 1):
            Gamma[(i - 1) * n : i * n, (j - 1) * m : j * m] = np.linalg.matrix_power(
                A, i - j
            ) @ B

    Omega = np.kron(np.eye(N, dtype=complex), Q)
    Omega[-q:, -q:] = P
    Psi = np.kron(np.eye(N, dtype=complex), R)

    term = C_big_r.conj().T @ Omega @ C_big_r + C_big_d.conj().T @ Psi @ C_big_d

    weight = INPUT_WEIGHT.lower()
    if weight == "delta":
        H = Gamma.conj().T @ term @ Gamma
        f = (
            2
            * z_k.conj().T
            @ Phi.conj().T
            @ (term - r_N.conj().T @ Omega @ C_big_r)
            @ Gamma
        )
    elif weight == "deltadot":
        H = Gamma.conj().T @ C_big_r.conj().T @ Omega @ C_big_r @ Gamma + Psi
        H = 2 * H
        f = 2 * (
            z_k.conj().T @ Phi.conj().T @ C_big_r.conj().T @ Omega @ C_big_r @ Gamma
            - r_N.conj().T @ Omega @ C_big_r @ Gamma
        )
    else:
        raise ValueError("Invalid INPUT_WEIGHT value")

    return H, f, Phi, Gamma
