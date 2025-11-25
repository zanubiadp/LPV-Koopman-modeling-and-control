"""
Construct matrix C given number of eigenvalues and states (MATLAB translation).
"""
from __future__ import annotations

import numpy as np
from collections.abc import Iterable


def build_C(nEig: int | Iterable[int], n_states: int) -> np.ndarray:
    """
    Build block-structured output matrix C.

    Args:
        nEig: Either total eigenvalues (int) or iterable with per-state counts.
        n_states: Number of states.

    Returns:
        C matrix of shape (n_states, sum(nEig)).
    """
    if np.isscalar(nEig):
        nEig_i = int(nEig) / n_states
        nEig = [int(nEig_i)] * n_states
    else:
        nEig = list(map(int, nEig))

    C = np.zeros((n_states, sum(nEig)), dtype=float)

    start_index = 0
    for i in range(n_states):
        end_index = start_index + nEig[i]
        C[i, start_index:end_index] = 1.0
        start_index = end_index

    return C
