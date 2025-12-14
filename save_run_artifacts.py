"""Utility to save and load run artifacts for replaying plots.

This module provides a small wrapper around pickle to store the
interesting objects produced by a run of `main.py`. We save a single
pickled file (default name `run_artifacts.pkl`) that `replay_plots.py`
can load to reproduce phase portraits and prediction plots without
re-running the learning pipeline.
"""
from __future__ import annotations

import pickle
from typing import Any


def save_run(path: str, artifacts: dict[str, Any]) -> None:
    """Save artifacts dict to `path` using pickle.

    Args:
        path: output filename (e.g. 'run_artifacts.pkl').
        artifacts: mapping of name -> object to persist.
    """
    with open(path, "wb") as fh:
        pickle.dump(artifacts, fh, protocol=pickle.HIGHEST_PROTOCOL)


def load_run(path: str) -> dict[str, Any]:
    """Load artifacts previously saved with :func:`save_run`.

    Returns the saved dict.
    """
    with open(path, "rb") as fh:
        return pickle.load(fh)
