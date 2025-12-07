"""
Apply a list of lifting (feature) functions to data columns.
"""
from __future__ import annotations

import numpy as np
from collections.abc import Callable
from typing import Iterable


def lifting_function(x: np.ndarray, phi: Iterable[Callable[[np.ndarray], np.ndarray]]) -> np.ndarray:
    """
    Evaluate each lifting function on the input data.

    Args:
        x: Input data (features in rows, samples in columns).
        phi: Iterable of callables; each should accept `x` and return a 1 x N array-like.

    Returns:
        Array of shape (len(phi), x.shape[1]) with lifted features.
    """
    x = np.asarray(x)
    phi_list = list(phi)
    # Use complex dtype so any imaginary parts from the lifting functions are preserved
    out = np.zeros((len(phi_list), x.shape[1]), dtype=complex)

    for ii, fn in enumerate(phi_list):
        # Ensure the returned values are treated as complex without discarding the imaginary part
        out[ii, :] = np.asarray(fn(x), dtype=complex).reshape(-1)

    return out
