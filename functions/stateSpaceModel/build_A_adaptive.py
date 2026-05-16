"""
Python translation of MATLAB's `build_A_adaptive`.
Learns eigenvalues from trajectory data via constrained optimization.
"""
from __future__ import annotations

import cma
import numpy as np
from typing import Callable, Iterable

from functions.stateSpaceModel.getCostGradientKordacc_re_adaptive import (  # type: ignore
    getCostGradientKordacc_re_adaptive,
)


def build_A_adaptive(
    Traj: Iterable,
    Traj_t: Iterable,
    data: np.ndarray,
    dt: float,
    nEig: int | Iterable[int],
    n_cc: int,
    rng: np.random.Generator | None = None,
    cost_fn: Callable | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Learn eigenvalues given trajectory data.

    Args:
        Traj: Iterable of trajectory data (supports varying lengths).
        Traj_t: Iterable of corresponding time vectors.
        data: Concatenated dataset matrix.
        dt: Simulation timestep.
        nEig: Number of eigenvalues (int or iterable; flattened for length).
        n_cc: Number of complex conjugate pairs.
        rng: Optional NumPy RNG for reproducible initialization.
        cost_fn: Optional override for cost/gradient function.

    Returns:
        lambdas_d: Discrete-time eigenvalues (column vector).
        lambdas_c: Continuous-time eigenvalues (column vector).
    """
    rng = rng or np.random.default_rng()
    n_eig_total = int(np.prod(nEig))

    cf = cost_fn or getCostGradientKordacc_re_adaptive
    cost_with_grad = lambda x: cf(x, Traj, Traj_t, data, n_cc)

    x0 = -10 + (10 - (-10)) * rng.random(n_eig_total)

    import os, logging
    log_path = f"cma_worker_{os.getpid()}.log"
    logging.basicConfig(filename=log_path, level=logging.INFO, format="%(asctime)s %(message)s")
    logging.info(f"CMA-ES started: n_eig={n_eig_total}")

    def logged_cost(x):
        cost, grad, _ = cost_with_grad(x)
        return cost

    es = cma.CMAEvolutionStrategy(x0, 5.0, {
        'maxiter': 5000,
        'verbose': 1,
        'bounds': [-15, 15],
    })

    while not es.stop():
        solutions = es.ask()
        fitvals = [logged_cost(x) for x in solutions]
        es.tell(solutions, fitvals)
        if es.result.iterations % 10 == 0:
            logging.info(f"iter={es.result.iterations} evals={es.result.evaluations} best={es.result.fbest:.6g}")

    x_opt = es.result.xbest
    logging.info(f"CMA-ES done: best={es.result.fbest:.6g}")

    # Reorder complex conjugate eigenvalues
    lambdas_list: list[complex] = []
    j = 1j
    for ii in range(0, 2 * n_cc, 2):
        lambdas_list.append(x_opt[ii] + j * x_opt[ii + 1])
        lambdas_list.append(x_opt[ii] - j * x_opt[ii + 1])

    for ii in range(2 * n_cc, len(x_opt)):
        lambdas_list.append(x_opt[ii])

    lambdas = np.array(lambdas_list, dtype=complex)
    lambdas_d = np.exp(dt * lambdas).reshape(-1, 1)
    lambdas_c = lambdas.reshape(-1, 1)

    return lambdas_d, lambdas_c
