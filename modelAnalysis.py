"""
System-theoretic analysis for 2D nonlinear systems with optional forcing

dot{x} = f(x, u),  x = [x1, x2]^T, u = scalar

Features:
- Single global switch USE_FORCING to choose:
    * free dynamics (u(t) ≡ u_eq, autonomous)
    * forced dynamics (u(t) time-varying)
- Symbolic:
    * Define f1(x1,x2,u), f2(x1,x2,u)  (EDIT THIS PART)
    * Fix an equilibrium input u_eq
    * Find equilibria for that u_eq
    * Compute Jacobians A = df/dx, B = df/du
- Numeric:
    * Define u_fun(t) based on USE_FORCING
    * Simulate nonlinear system
    * Linearise around (x_eq, u_eq) and simulate linear system
    * Classify equilibria from A
    * Plots (titles change with mode):
        - Phase portrait
        - Nonlinear time series
        - Nonlinear vs linearised (x1, x2 in two subplots)

Dependencies:
    pip install numpy scipy sympy matplotlib
"""

import numpy as np
import sympy as sp
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


# ============================================================
# GLOBAL SWITCH: FREE vs FORCED
# ============================================================

USE_FORCING = True   # <- set to True for forced simulations, False for free


# ============================================================
# 1. SYMBOLIC DEFINITION OF THE SYSTEM
# ============================================================

def get_symbolic_system():
    """
    Define the symbolic 2D system with scalar input u:

        dot{x1} = f1(x1, x2, u)
        dot{x2} = f2(x1, x2, u)

    EDIT THIS FUNCTION to change the system.
    """
    x1, x2, u = sp.symbols('x1 x2 u')

    # Example: Non affine Van der Pol with input
    #   dot x1 = x2
    #   dot x2 = mu (1 - x1^2) x2 - x1 + (1 + a x1^2) * tanh(b u)
    mu = 1.0
    a = 0.5
    b = 1.0

    f1 = x2
    f2 = mu * (1 - x1**2) * x2 - x1 + (1 + a * x1**2) * sp.tanh(b * u)

    # Alternative: comment above, write your own f1, f2(x1,x2,u) here.

    return x1, x2, u, f1, f2


# ============================================================
# 2. EQUILIBRIA AND JACOBIANS
# ============================================================

def find_equilibria(x1, x2, u, f1, f2, u_eq=0.0):
    """
    Solve f(x, u_eq) = 0 for equilibria:

        f1(x1,x2,u_eq) = 0
        f2(x1,x2,u_eq) = 0

    Returns list of (x1_eq, x2_eq) as sympy values.
    """
    f1_eq = f1.subs(u, u_eq)
    f2_eq = f2.subs(u, u_eq)
    eqs = sp.solve([sp.Eq(f1_eq, 0), sp.Eq(f2_eq, 0)], (x1, x2), dict=True)
    equilibria = []
    for sol in eqs:
        equilibria.append((sp.simplify(sol[x1]), sp.simplify(sol[x2])))
    return equilibria


def get_jacobians(x1, x2, u, f1, f2):
    """
    Return symbolic Jacobians:

        Jx = df/dx  (2x2)
        Ju = df/du  (2x1)
    """
    f_vec = sp.Matrix([f1, f2])
    Jx = f_vec.jacobian([x1, x2])
    Ju = f_vec.jacobian([u])
    return Jx, Ju


def linearize_at_equilibrium(Jx_sym, Ju_sym, x1, x2, u, eq_point, u_eq):
    """
    Evaluate Jacobians at (x_eq, u_eq) and return numeric A, B.
    """
    x1e, x2e = eq_point
    subs_dict = {x1: x1e, x2: x2e, u: u_eq}

    A = np.array(Jx_sym.subs(subs_dict), dtype=float)
    B = np.array(Ju_sym.subs(subs_dict), dtype=float).reshape(2, 1)

    return A, B


# ============================================================
# 3. NUMERIC SYSTEM WITH INPUT
# ============================================================

def make_numeric_system(f1, f2, x1, x2, u, u_fun):
    """
    Convert sympy expressions to a numeric function f(t,x) that uses u_fun(t).

    Returns: f_numeric(t, x) -> np.array([dx1, dx2])
    """
    f1_func = sp.lambdify((x1, x2, u), f1, 'numpy')
    f2_func = sp.lambdify((x1, x2, u), f2, 'numpy')

    def f_numeric(t, state):
        x1_val, x2_val = state
        u_val = u_fun(t)
        return np.array([
            f1_func(x1_val, x2_val, u_val),
            f2_func(x1_val, x2_val, u_val)
        ], dtype=float)

    return f_numeric


# ============================================================
# 4. EIGENVALUE-BASED CLASSIFICATION
# ============================================================

def classify_equilibrium(A, tol=1e-8):
    """
    Classify equilibrium based on eigenvalues of A.
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
            return 'degenerate / non hyperbolic'
    else:
        if np.all(np.abs(real_parts) < tol):
            return 'center (linear), nonlinear terms decide'
        elif np.all(real_parts < -tol):
            return 'stable focus (spiral)'
        elif np.all(real_parts > tol):
            return 'unstable focus (spiral)'
        else:
            return 'focus with mixed real parts, exotic in 2D'


# ============================================================
# 5. PLOTTING
# ============================================================

def plot_phase_portrait(f_numeric, mode_label,
                        xlim=(-3, 3), ylim=(-3, 3),
                        n_grid=20, trajectories=None, t_span=(0, 10)):
    """
    Plot vector field at t=0 and trajectories.
    mode_label: 'free' or 'forced' (used in title).
    """
    if trajectories is None:
        trajectories = [[-2, 0], [-1, 1], [1.5, 0.5], [2, -1]]

    x1_vals = np.linspace(xlim[0], xlim[1], n_grid)
    x2_vals = np.linspace(ylim[0], ylim[1], n_grid)
    X1, X2 = np.meshgrid(x1_vals, x2_vals)

    U = np.zeros_like(X1)
    V = np.zeros_like(X2)

    # Vector field at t=0
    for i in range(n_grid):
        for j in range(n_grid):
            dx = f_numeric(0.0, [X1[i, j], X2[i, j]])
            U[i, j] = dx[0]
            V[i, j] = dx[1]

    plt.figure(figsize=(7, 6))
    plt.streamplot(X1, X2, U, V, density=1.0, linewidth=0.8, arrowsize=1)
    plt.xlabel('x1')
    plt.ylabel('x2')

    if mode_label == 'free':
        title = 'Phase portrait (free dynamics, u(t) ≡ u_eq)'
    else:
        title = 'Phase portrait (forced dynamics, vector field at t=0)'
    plt.title(title)

    # Trajectories
    for x0 in trajectories:
        sol = solve_ivp(f_numeric, t_span, x0, dense_output=True, max_step=0.02)
        plt.plot(sol.y[0, :], sol.y[1, :], lw=1.5, label=f'x0={x0}')

    plt.legend()
    plt.grid(True)
    plt.tight_layout()


def plot_time_series(f_numeric, x0, mode_label,
                     t_span=(0, 10), n_points=1000):
    """
    Plot x1(t), x2(t) for the nonlinear system.
    """
    t_eval = np.linspace(t_span[0], t_span[1], n_points)
    sol = solve_ivp(f_numeric, t_span, x0, t_eval=t_eval, max_step=0.02)

    plt.figure(figsize=(7, 4))
    plt.plot(sol.t, sol.y[0, :], label='x1 nonlinear')
    plt.plot(sol.t, sol.y[1, :], label='x2 nonlinear')
    plt.xlabel('t')
    plt.ylabel('state')

    if mode_label == 'free':
        title = f'Free nonlinear time series from x0 = {x0}'
    else:
        title = f'Forced nonlinear time series from x0 = {x0}'
    plt.title(title)

    plt.grid(True)
    plt.legend()
    plt.tight_layout()


def plot_linear_vs_nonlinear(f_numeric, A, B, eq_point,
                             u_fun, u_eq, x0, mode_label,
                             t_span=(0, 10), n_points=1000):
    """
    Compare nonlinear and linearised system from same initial condition.

    Nonlinear:   dot{x} = f(x, u(t))
    Linearised:  dot{x} = A (x - x*) + B (u(t) - u_eq)
    """
    x_star = np.array([float(eq_point[0]), float(eq_point[1])], dtype=float)
    t_eval = np.linspace(t_span[0], t_span[1], n_points)

    # Nonlinear
    sol_nl = solve_ivp(f_numeric, t_span, x0, t_eval=t_eval, max_step=0.02)

    # Linear
    def f_linear(t, x):
        u_val = u_fun(t)
        return A @ (x - x_star) + B.flatten() * (u_val - u_eq)

    sol_lin = solve_ivp(f_linear, t_span, x0, t_eval=t_eval, max_step=0.02)

    plt.figure(figsize=(8, 6))

    # x1
    ax1 = plt.subplot(2, 1, 1)
    ax1.plot(sol_nl.t, sol_nl.y[0, :], label='x1 nonlinear')
    ax1.plot(sol_lin.t, sol_lin.y[0, :], '--', label='x1 linearised')
    ax1.set_ylabel('x1')

    if mode_label == 'free':
        title = f'Free nonlinear vs linearised, x* = {x_star}, x0 = {x0}'
    else:
        title = f'Forced nonlinear vs linearised, x* = {x_star}, x0 = {x0}'
    ax1.set_title(title)

    ax1.grid(True)
    ax1.legend()

    # x2
    ax2 = plt.subplot(2, 1, 2, sharex=ax1)
    ax2.plot(sol_nl.t, sol_nl.y[1, :], label='x2 nonlinear')
    ax2.plot(sol_lin.t, sol_lin.y[1, :], '--', label='x2 linearised')
    ax2.set_xlabel('t')
    ax2.set_ylabel('x2')
    ax2.grid(True)
    ax2.legend()

    plt.tight_layout()


# ============================================================
# 6. MAIN
# ============================================================

def main():
    mode_label = 'forced' if USE_FORCING else 'free'

    # Symbolic system
    x1, x2, u, f1, f2 = get_symbolic_system()
    print("System:")
    print("  dot x1 =", f1)
    print("  dot x2 =", f2)
    print()

    # Equilibrium input
    u_eq = 0.0

    # Equilibria at u = u_eq
    equilibria = find_equilibria(x1, x2, u, f1, f2, u_eq=u_eq)
    print(f"Equilibria for u_eq = {u_eq}:")
    for k, eq in enumerate(equilibria):
        print(f"  Eq {k}: x* = ({eq[0]}, {eq[1]})")
    print()

    # Jacobians
    Jx, Ju = get_jacobians(x1, x2, u, f1, f2)
    print("Symbolic Jx = df/dx :")
    sp.pprint(Jx)
    print("\nSymbolic Ju = df/du :")
    sp.pprint(Ju)
    print()

    # Input function depending on mode
    if USE_FORCING:
        # Forced case: time-varying input
        def u_fun(t):
            return 0.5 * np.sin(1.0 * t)     # edit as you like
    else:
        # Free case: constant input u(t) ≡ u_eq
        def u_fun(t):
            return u_eq

    # Numeric system
    f_numeric = make_numeric_system(f1, f2, x1, x2, u, u_fun)

    # Linearisation and classification
    print("Linearisation at each equilibrium:")
    A_list = []
    B_list = []

    for k, eq in enumerate(equilibria):
        A, B = linearize_at_equilibrium(Jx, Ju, x1, x2, u, eq, u_eq)
        A_list.append(A)
        B_list.append(B)

        eigvals, _ = np.linalg.eig(A)
        eq_type = classify_equilibrium(A)

        print(f"Equilibrium {k}: x* = {eq}")
        print("  A =")
        print(A)
        print("  B =")
        print(B)
        print("  Eigenvalues =", eigvals)
        print("  Type:", eq_type)
        print()

    # Phase portrait
    plot_phase_portrait(
        f_numeric,
        mode_label=mode_label,
        xlim=(-3, 3),
        ylim=(-3, 3),
        n_grid=25,
        trajectories=[[-2, 0], [-1, 1], [1.5, 0.5], [2, -1]],
        t_span=(0, 10)
    )

    # Nonlinear time series
    x0_ts = [1.0, 0.0]
    plot_time_series(
        f_numeric,
        x0=x0_ts,
        mode_label=mode_label,
        t_span=(0, 10)
    )

    # Nonlinear vs linearised
    if len(equilibria) > 0:
        eq0 = equilibria[0]
        A0 = A_list[0]
        B0 = B_list[0]

        # Initial condition near equilibrium
        x0_compare = [float(eq0[0]) + 0.5, float(eq0[1])]

        plot_linear_vs_nonlinear(
            f_numeric,
            A=A0,
            B=B0,
            eq_point=eq0,
            u_fun=u_fun,
            u_eq=u_eq,
            x0=x0_compare,
            mode_label=mode_label,
            t_span=(0, 10),
            n_points=1000
        )
    else:
        print("No equilibria found at u_eq, skipping linear vs nonlinear comparison.")

    plt.show()


if __name__ == "__main__":
    main()
