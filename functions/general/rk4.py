from __future__ import annotations

import numpy as np

#def f_u_numba(mu_val: float, a_val: float, b_val: float, x: np.ndarray, u: float) -> np.ndarray:
def f_u_numba(x: np.ndarray, u: float) -> np.ndarray:
    x1 = x[0]
    x2 = x[1]
    x3 = x[2]
    f1 = 2*x2
    #f2 = mu_val * (1.0 - x1 * x1) * x2 - x1 + (1.0 + a_val * x1 * x1) * np.tanh(b_val * x3)
    f2 = -0.8 * x1 + 2 * x2 - 10 * x2 * x1 ** 2 + x3
    f3 = u
    return np.array([f1, f2, f3], dtype=np.float64)


def rk4_step_numba(x: np.ndarray, u: float, dt_val: float) -> np.ndarray:
    k1 = f_u_numba(x, u)
    k2 = f_u_numba(x + k1 * dt_val / 2.0, u)
    k3 = f_u_numba(x + k2 * dt_val / 2.0, u)
    k4 = f_u_numba(x + k3 * dt_val, u)
    return x + (dt_val / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def rk4_step_linear(Z: np.ndarray, A: np.ndarray, B: np.ndarray, u: float, dt: float) -> np.ndarray:
    """Runge-Kutta 4 integrator for linear state-space system."""
    def f_lin(vec: np.ndarray) -> np.ndarray:
        return A @ vec + B * u

    Y1 = f_lin(Z)
    Y2 = f_lin(Z + Y1 * dt / 2.0)
    Y3 = f_lin(Z + Y2 * dt / 2.0)
    Y4 = f_lin(Z + Y3 * dt)
    return Z + (dt / 6.0) * (Y1 + 2 * Y2 + 2 * Y3 + Y4)
