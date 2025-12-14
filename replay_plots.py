"""Load saved run artifacts and reproduce plots without re-running learning.

Usage:
    python3 replay_plots.py run_artifacts.pkl

If no filename is provided the script looks for `run_artifacts.pkl` in the
current working directory.
"""
from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import LinearNDInterpolator

from save_run_artifacts import load_run


def make_phi_hat(points: np.ndarray, phi_vals: np.ndarray):
    """Recreate callable interpolants from saved points and values.

    points: shape (n_samples, ndim)
    phi_vals: shape (n_features, n_samples)
    Returns list of callables that accept a 1-D point (ndim,) or array-like.
    """
    phi_hat = []
    for ii in range(phi_vals.shape[0]):
        interp = LinearNDInterpolator(points, phi_vals[ii, :])

        def make_fn(interp):
            def fn(x: np.ndarray) -> np.ndarray:
                x_arr = np.asarray(x, dtype=float)
                vals = interp(x_arr.T)
                return np.asarray(vals).reshape(-1)

            return fn

        phi_hat.append(make_fn(interp))
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
    path = sys.argv[1] if len(sys.argv) > 1 else "run_artifacts.pkl"
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Artifacts file not found: {p.resolve()}")

    artifacts = load_run(str(p))

    # Reconstruct interpolants if needed
    points = np.asarray(artifacts["interp_points"], dtype=float)
    phi_vals = np.asarray(artifacts["phi_vals"])
    phi_hat = make_phi_hat(points, phi_vals)

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
