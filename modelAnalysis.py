"""
General system-theoretic analysis for 2D autonomous systems

dot{x} = f(x),  x = [x1, x2]^T

What it does:
- Symbolic:
    * Define f1(x1,x2), f2(x1,x2)  (EDIT THIS PART)
    * Find equilibria
    * Compute Jacobian
- Numeric:
    * Linearize at each equilibrium (A = df/dx)
    * Eigenvalues and qualitative classification
    * Phase portrait (vector field + trajectories)
    * Time series for one trajectory
    * Comparison of nonlinear vs linearised system from same x0

Dependencies:
    pip install numpy scipy sympy matplotlib
"""

import numpy as np
import sympy as sp
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


# ============================================================
# 1. SYMBOLIC DEFINITION OF THE SYSTEM
# ============================================================

def get_symbolic_system():
    """
    Define the symbolic 2D system:
        dot{x1} = f1(x1, x2)
        dot{x2} = f2(x1, x2)

    EDIT THIS FUNCTION to change the system.
    """
    x1, x2 = sp.symbols('x1 x2')

    # Example: Van der Pol type, u = 0, mu = 1
    mu = 1.0
    f1 = x2
    f2 = mu * (1 - x1**2) * x2 - x1

    # Example alternativ0 (comment above, uncomment this):
    # alpha, beta, delta = 1.0, 1.0, 0.2
    # f1 = x2
    # f2 = -delta*x2 - alpha*x1 - beta*x1**3

    return x1, x2, f1, f2


# ============================================================
# 2. FIND EQUILIBRIA & JACOBIAN SYMBOLICALLY
# ============================================================

def find_equilibria(x1, x2, f1, f2):
    """
    Solve f1(x1,x2) = 0, f2(x1,x2) = 0 symbolically.
    Returns a list of (x1_eq, x2_eq) as sympy values.
    """
    eqs = sp.solve([sp.Eq(f1, 0), sp.Eq(f2, 0)], (x1, x2), dict=True)
    equilibria = []
    for sol in eqs:
        equilibria.append((sp.simplify(sol[x1]), sp.simplify(sol[x2])))
    return equilibria


def get_jacobian(x1, x2, f1, f2):
    """
    Return the symbolic Jacobian matrix df/dx evaluated symbolically.
    """
    J = sp.Matrix([f1, f2]).jacobian([x1, x2])
    return J


# ============================================================
# 3. NUMERIC VERSION OF THE SYSTEM AND LINEARIZATION
# ============================================================

def make_numeric_system(f1, f2, x1, x2):
    """
    Convert sympy expressions to a numerical function f(x) for solve_ivp.

    Returns: f_numeric(t, x) -> np.array([dx1, dx2])
    """
    f1_func = sp.lambdify((x1, x2), f1, 'numpy')
    f2_func = sp.lambdify((x1, x2), f2, 'numpy')

    def f_numeric(t, state):
        x1_val, x2_val = state
        return np.array([f1_func(x1_val, x2_val),
                         f2_func(x1_val, x2_val)], dtype=float)

    return f_numeric


def linearize_at_equilibrium(J_symbolic, x1, x2, eq_point):
    """
    Evaluate the symbolic Jacobian at a given equilibrium (x1e, x2e) and
    return the numeric A matrix.
    """
    x1e, x2e = eq_point
    A = np.array(J_symbolic.subs({x1: x1e, x2: x2e})).astype(np.float64)
    return A


# ============================================================
# 4. EIGENVALUE-BASED CLASSIFICATION
# ============================================================

def classify_equilibrium(A, tol=1e-8):
    """
    Classify equilibrium based on eigenvalues of A.

    Returns a string with a qualitative type, e.g.:
    - 'stable node'
    - 'unstable node'
    - 'saddle'
    - 'stable focus (spiral)'
    - 'unstable focus (spiral)'
    - 'center (linear)'
    """
    eigvals, _ = np.linalg.eig(A)
    real_parts = eigvals.real
    imag_parts = eigvals.imag

    has_pos_real = np.any(real_parts > tol)
    has_neg_real = np.any(real_parts < -tol)
    all_real = np.all(np.abs(imag_parts) < tol)

    if has_pos_real and has_neg_real:
        return 'saddle (unstable)'

    if all_real:
        if np.all(real_parts < -tol):
            return 'stable node'
        elif np.all(real_parts > tol):
            return 'unstable node'
        else:
            return 'degenerate / non-hyperbolic (needs nonlinear analysis)'
    else:
        if np.all(np.abs(real_parts) < tol):
            return 'center (linear), nonlinear terms decide'
        elif np.all(real_parts < -tol):
            return 'stable focus (spiral)'
        elif np.all(real_parts > tol):
            return 'unstable focus (spiral)'
        else:
            return 'focus with mixed sign real parts (weird in 2D real)'


# ============================================================
# 5. PLOTTING: PHASE PORTRAIT AND TIME SERIES
# ============================================================

def plot_phase_portrait(f_numeric, xlim=(-3, 3), ylim=(-3, 3),
                        n_grid=20, trajectories=None, t_span=(0, 10)):
    """
    Plot vector field and trajectories of the system.
    """
    if trajectories is None:
        trajectories = [[-2, 0], [-1, 1], [1, 1], [2, -1]]

    x1_vals = np.linspace(xlim[0], xlim[1], n_grid)
    x2_vals = np.linspace(ylim[0], ylim[1], n_grid)
    X1, X2 = np.meshgrid(x1_vals, x2_vals)

    U = np.zeros_like(X1)
    V = np.zeros_like(X2)

    for i in range(n_grid):
        for j in range(n_grid):
            dx = f_numeric(0, [X1[i, j], X2[i, j]])
            U[i, j] = dx[0]
            V[i, j] = dx[1]

    plt.figure(figsize=(7, 6))
    plt.streamplot(X1, X2, U, V, density=1.0, linewidth=0.8, arrowsize=1)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title('Phase portrait: vector field & trajectories')

    for x0 in trajectories:
        sol = solve_ivp(f_numeric, t_span, x0, dense_output=True, max_step=0.05)
        plt.plot(sol.y[0, :], sol.y[1, :], lw=1.5, label=f'x0={x0}')

    plt.legend()
    plt.grid(True)
    plt.tight_layout()


def plot_time_series(f_numeric, x0, t_span=(0, 10), n_points=1000):
    """
    Plot x1(t), x2(t) for a single initial condition (nonlinear system).
    """
    t_eval = np.linspace(t_span[0], t_span[1], n_points)
    sol = solve_ivp(f_numeric, t_span, x0, t_eval=t_eval, max_step=0.05)

    plt.figure(figsize=(7, 4))
    plt.plot(sol.t, sol.y[0, :], label='x1(t) nonlinear')
    plt.plot(sol.t, sol.y[1, :], label='x2(t) nonlinear')
    plt.xlabel('t')
    plt.ylabel('state')
    plt.title(f'Nonlinear time series from x0 = {x0}')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()


# ============================================================
# 6. NONLINEAR VS LINEARISED COMPARISON
# ============================================================

def plot_linear_vs_nonlinear(f_numeric, A, eq_point, x0,
                             t_span=(0, 10), n_points=1000):
    """
    Compare nonlinear system and linearised system from same initial condition.

    Nonlinear: dot{x} = f(x)
    Linearised around x*:

        dot{x} = A (x - x*)

    Simulate x(t) for both. Two subplots:
      - top: x1 nonlinear vs linear
      - bottom: x2 nonlinear vs linear
    """
    x_star = np.array([float(eq_point[0]), float(eq_point[1])], dtype=float)

    t_eval = np.linspace(t_span[0], t_span[1], n_points)

    # Nonlinear simulation
    sol_nl = solve_ivp(f_numeric, t_span, x0, t_eval=t_eval, max_step=0.05)

    # Linearised system
    def f_linear(t, x):
        return A @ (x - x_star)

    sol_lin = solve_ivp(f_linear, t_span, x0, t_eval=t_eval, max_step=0.05)

    plt.figure(figsize=(8, 6))

    # x1 subplot
    ax1 = plt.subplot(2, 1, 1)
    ax1.plot(sol_nl.t, sol_nl.y[0, :], label='x1 nonlinear')
    ax1.plot(sol_lin.t, sol_lin.y[0, :], '--', label='x1 linearised')
    ax1.set_ylabel('x1')
    ax1.set_title(f'Nonlinear vs linearised (x* = {x_star}, x0 = {x0})')
    ax1.grid(True)
    ax1.legend()

    # x2 subplot
    ax2 = plt.subplot(2, 1, 2, sharex=ax1)
    ax2.plot(sol_nl.t, sol_nl.y[1, :], label='x2 nonlinear')
    ax2.plot(sol_lin.t, sol_lin.y[1, :], '--', label='x2 linearised')
    ax2.set_xlabel('t')
    ax2.set_ylabel('x2')
    ax2.grid(True)
    ax2.legend()

    plt.tight_layout()


# ============================================================
# 7. MAIN: RUN FULL ANALYSIS
# ============================================================

def main():
    # Symbolic system
    x1, x2, f1, f2 = get_symbolic_system()
    print("System:")
    print("  dot x1 =", f1)
    print("  dot x2 =", f2)
    print()

    # Equilibria
    equilibria = find_equilibria(x1, x2, f1, f2)
    print("Equilibria (symbolic):")
    for k, eq in enumerate(equilibria):
        print(f"  Eq {k}: x* = ({eq[0]}, {eq[1]})")
    print()

    # Jacobian
    J = get_jacobian(x1, x2, f1, f2)
    print("Symbolic Jacobian J(x):")
    sp.pprint(J)
    print()

    # Numeric system
    f_numeric = make_numeric_system(f1, f2, x1, x2)

    # Linearisation & classification
    print("Linearization at each equilibrium:")
    A_mats = []
    for k, eq in enumerate(equilibria):
        A = linearize_at_equilibrium(J, x1, x2, eq)
        eigvals, _ = np.linalg.eig(A)
        eq_type = classify_equilibrium(A)

        A_mats.append(A)

        print(f"Equilibrium {k}: x* = {eq}")
        print("  A =")
        print(A)
        print("  Eigenvalues =", eigvals)
        print("  Type:", eq_type)
        print()

    # Phase portrait
    plot_phase_portrait(
        f_numeric,
        xlim=(-3, 3),
        ylim=(-3, 3),
        n_grid=25,
        trajectories=[[-2, 0], [-1, 1], [1.5, 0.5], [2, -1]],
        t_span=(0, 10)
    )

    # Nonlinear time series
    x0_ts = [1.0, 0.0]
    plot_time_series(f_numeric, x0=x0_ts, t_span=(0, 10))

    # Nonlinear vs linearised comparison
    if len(equilibria) > 0:
        eq0 = equilibria[0]
        A0 = A_mats[0]

        # perturb equilibrium a bit
        x0_compare = [float(eq0[0]) + 0.5, float(eq0[1])]

        plot_linear_vs_nonlinear(
            f_numeric,
            A=A0,
            eq_point=eq0,
            x0=x0_compare,
            t_span=(0, 10),
            n_points=1000
        )
    else:
        print("No equilibria found, skipping linear vs nonlinear comparison.")

    plt.show()


if __name__ == "__main__":
    main()
