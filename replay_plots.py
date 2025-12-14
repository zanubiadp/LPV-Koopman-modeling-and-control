"""
Load saved run artifacts and reproduce plots without re-running learning.

Usage:
    python3 replay_plots.py run_artifacts.pkl
    python3 replay_plots.py run_artifacts.pkl --rebuild-interp
    python3 replay_plots.py run_artifacts.pkl --rebuild-interp --debug

If no filename is provided the script looks for `run_artifacts.pkl` in the
current working directory.

Notes:
- Rebuilding LinearNDInterpolator can be very slow. This script skips it by
  default because it is not used for plotting in this file.
- If you REALLY need interpolants for something else, pass --rebuild-interp.
"""

from __future__ import annotations

import sys
from pathlib import Path
import time

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import LinearNDInterpolator
from scipy.spatial import Delaunay

from save_run_artifacts import load_run


def make_phi_hat(points: np.ndarray, phi_vals: np.ndarray, debug: bool = False):
    """Recreate callable interpolants from saved points and values.

    points: shape (n_samples, 3)   (your case: ndim=3)
    phi_vals: shape (n_features, n_samples)

    Returns list of callables that accept:
      - a single point shape (3,)
      - or an array of points shape (n_pts, 3)
    and returns a 1D array of interpolated values (n_pts,).
    """
    points = np.asarray(points, dtype=float)
    phi_vals = np.asarray(phi_vals, dtype=float)

    if points.ndim != 2:
        raise ValueError(f"`points` must be 2D (n_samples, ndim). Got {points.shape}")
    if phi_vals.ndim != 2:
        raise ValueError(f"`phi_vals` must be 2D (n_features, n_samples). Got {phi_vals.shape}")
    if points.shape[0] != phi_vals.shape[1]:
        raise ValueError(
            f"Mismatch: points has {points.shape[0]} samples but phi_vals has {phi_vals.shape[1]}."
        )

    # Expensive step: build triangulation ONCE, reuse it for each feature.
    t0 = time.perf_counter()
    tri = Delaunay(points)
    if debug:
        print(f"[debug] Delaunay built in {time.perf_counter() - t0:.3f} s")

    phi_hat = []
    for ii in range(phi_vals.shape[0]):
        interp = LinearNDInterpolator(tri, phi_vals[ii, :])

        # default-arg capture so each fn uses its own interp
        def fn(x, interp=interp):
            x_arr = np.asarray(x, dtype=float)

            # Accept (3,) or (n,3)
            if x_arr.ndim == 1:
                if x_arr.shape[0] != points.shape[1]:
                    raise ValueError(f"Point must have shape ({points.shape[1]},), got {x_arr.shape}")
                vals = interp(x_arr)  # scalar
                return np.atleast_1d(vals).astype(float)
            elif x_arr.ndim == 2:
                if x_arr.shape[1] != points.shape[1]:
                    raise ValueError(f"Points must have shape (n,{points.shape[1]}), got {x_arr.shape}")
                vals = interp(x_arr)  # (n,)
                return np.asarray(vals, dtype=float).reshape(-1)
            else:
                raise ValueError(f"x must be 1D or 2D, got shape {x_arr.shape}")

        phi_hat.append(fn)

    return phi_hat


def plot_phase_portrait(Traj: list, ax=None):
    if ax is None:
        fig, ax = plt.subplots()

    for row in Traj:
        for traj in row:
            if traj is None:
                continue
            ax.plot(traj[0, :], traj[1, :], color=(0.7, 0.7, 0.7), linewidth=0.8)

    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")
    ax.set_title("Phase portrait (x1,x2)")
    ax.grid(True)
    ax.set_aspect("equal")
    return ax


def plot_predictions(predictions: list):
    for pred in predictions:
        Xtrue = pred["Xtrue"]
        X_gra = pred["X_gra"]
        Utrue = pred["Utrue"]
        title = pred.get("title", "Prediction")

        fig_pred, axes = plt.subplots(3, 1, figsize=(8, 8))
        N = Xtrue.shape[1]
        time_ticks = np.linspace(0, N - 1, 5)

        axes[0].plot(Xtrue[0, :], linewidth=2, color="#001BFF", label=r"$x_1$")
        axes[0].plot(X_gra[0, :], linewidth=2, linestyle=":", color="magenta", label=r"$\hat x_1$")
        axes[0].legend(fontsize=12)
        axes[0].grid(True)
        axes[0].set_xticks(time_ticks)
        axes[0].set_xticklabels([f"{t:.2f}" for t in time_ticks])
        axes[0].set_ylabel(r"$x_1$")

        axes[1].plot(Xtrue[1, :], linewidth=2, color="#001BFF", label=r"$x_2$")
        axes[1].plot(X_gra[1, :], linewidth=2, linestyle=":", color="magenta", label=r"$\hat x_2$")
        axes[1].legend(fontsize=12)
        axes[1].grid(True)
        axes[1].set_xticks(time_ticks)
        axes[1].set_xticklabels([f"{t:.2f}" for t in time_ticks])
        axes[1].set_ylabel(r"$x_2$")

        axes[2].plot(Utrue, linewidth=2, color="#001BFF", label=r"$u$")
        axes[2].grid(True)
        axes[2].set_xticks(time_ticks)
        axes[2].set_xticklabels([f"{t:.2f}" for t in time_ticks])
        axes[2].set_xlabel("time [samples]")
        axes[2].set_ylabel(r"$u$")

        fig_pred.suptitle(title)
        plt.tight_layout()


def main():
    args = sys.argv[1:]

    # crude CLI parsing (because apparently we love simplicity until we don't)
    rebuild_interp = "--rebuild-interp" in args
    debug = "--debug" in args
    args = [a for a in args if a not in ("--rebuild-interp", "--debug")]

    path = args[0] if len(args) > 0 else "run_artifacts.pkl"
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Artifacts file not found: {p.resolve()}")

    t0 = time.perf_counter()
    artifacts = load_run(str(p))
    if debug:
        print(f"[debug] Loaded artifacts in {time.perf_counter() - t0:.3f} s")

    # Reconstruct interpolants ONLY if explicitly requested.
    # In this script, phi_hat is not used for plotting, so default is OFF.
    if rebuild_interp:
        points = np.asarray(artifacts["interp_points"], dtype=float)
        phi_vals = np.asarray(artifacts["phi_vals"], dtype=float)

        if debug:
            print(f"[debug] points.shape = {points.shape}, phi_vals.shape = {phi_vals.shape}")

        t1 = time.perf_counter()
        phi_hat = make_phi_hat(points, phi_vals, debug=debug)
        if debug:
            print(f"[debug] Built {len(phi_hat)} interpolants in {time.perf_counter() - t1:.3f} s")

    # Phase portrait
    fig, ax = plt.subplots(figsize=(6, 6))
    plot_phase_portrait(artifacts["Traj"], ax=ax)

    # Replay predictions
    preds = artifacts.get("predictions", [])
    if preds:
        plot_predictions(preds)

    plt.show()


if __name__ == "__main__":
    main()
