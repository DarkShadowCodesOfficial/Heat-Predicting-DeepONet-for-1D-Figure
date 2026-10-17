"""Finite-element reference solvers (ground truth).

`run_fem` and `run_fem_nonlinear` are the notebook's solvers. Two things differ,
neither of which changes the numerical result:

* the solvers stop as soon as the last requested time has been recorded instead
  of always marching to ``MAX_TIME`` (so asking for t=60 s costs 60 s of steps);
* the Picard stiffness assembly is vectorised instead of looped in Python.

The linear solver is still a dense LU factorisation, like the notebook, so the
speed-ups reported by the demo are relative to *this* FEM implementation.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import lu_factor, lu_solve

from .config import (
    BETA_NL, H, HEATER, MAX_TIME, NUM_NODES, SENSOR_X, T_REF, TIME_STEP, X_NODES,
)

# --------------------------------------------------------------------------- #
# Mass / stiffness matrices (assembled once)
# --------------------------------------------------------------------------- #
M_LOCAL = (H / 6.0) * np.array([[2.0, 1.0], [1.0, 2.0]])
K_LOCAL = (1.0 / H) * np.array([[1.0, -1.0], [-1.0, 1.0]])


def _assemble_constant() -> tuple[np.ndarray, np.ndarray]:
    M = np.zeros((NUM_NODES, NUM_NODES))
    K = np.zeros((NUM_NODES, NUM_NODES))
    for e in range(NUM_NODES - 1):
        nodes = [e, e + 1]
        for a in range(2):
            for b in range(2):
                M[nodes[a], nodes[b]] += M_LOCAL[a, b]
                K[nodes[a], nodes[b]] += K_LOCAL[a, b]
    return M, K


M, K = _assemble_constant()


def initial_field(ic_vals) -> np.ndarray:
    """Interpolate the sensor values onto the FEM nodes and pin the heater."""
    T = np.interp(X_NODES, SENSOR_X, np.asarray(ic_vals, dtype=float))
    T[0] = HEATER
    return T


def _n_steps() -> tuple[int, float]:
    n_dt = int(MAX_TIME / TIME_STEP)
    return n_dt, MAX_TIME / n_dt


# --------------------------------------------------------------------------- #
# Linear FEM  (constant alpha)
# --------------------------------------------------------------------------- #
def run_fem(ic_vals, alpha: float, times) -> dict[float, np.ndarray]:
    """Backward-Euler FEM for dT/dt = alpha d2T/dx2.

    Dirichlet T(0,t)=HEATER, natural (insulated) boundary at x=L.
    Returns ``{t: T(x_nodes, t)}`` for each requested time.
    """
    n_dt, dt = _n_steps()
    A = M + alpha * dt * K
    A[0, :] = 0.0
    A[0, 0] = 1.0
    A_factored = lu_factor(A)

    T = initial_field(ic_vals)

    times = sorted(times)
    results: dict[float, np.ndarray] = {}
    t_current = 0.0
    next_idx = 0
    if times and times[0] <= 1e-9:
        results[times[0]] = T.copy()
        next_idx = 1

    for _ in range(n_dt):
        if next_idx >= len(times):
            break                                   # nothing left to record
        rhs = M @ T
        rhs[0] = HEATER
        T = lu_solve(A_factored, rhs)
        t_current += dt
        while next_idx < len(times) and t_current >= times[next_idx] - 1e-6:
            results[times[next_idx]] = T.copy()
            next_idx += 1

    return results


# --------------------------------------------------------------------------- #
# Nonlinear FEM  (alpha depends on T, Picard iteration)
# --------------------------------------------------------------------------- #
def alpha_of_T(T, alpha0: float, beta: float, T_ref: float = T_REF):
    """alpha(T) = alpha0 * (1 + beta * (T - T_ref))."""
    return alpha0 * (1.0 + beta * (T - T_ref))


def assemble_K_nonlinear(T_field, alpha0, beta, T_ref=T_REF) -> np.ndarray:
    """Stiffness matrix with one alpha per element (evaluated at its mean T)."""
    T_avg = 0.5 * (T_field[:-1] + T_field[1:])
    k_e = alpha_of_T(T_avg, alpha0, beta, T_ref) / H        # (n_el,)
    idx = np.arange(NUM_NODES - 1)
    K_nl = np.zeros((NUM_NODES, NUM_NODES))
    K_nl[idx, idx] += k_e
    K_nl[idx + 1, idx + 1] += k_e
    K_nl[idx, idx + 1] -= k_e
    K_nl[idx + 1, idx] -= k_e
    return K_nl


def run_fem_nonlinear(ic_vals, alpha0, beta, times, T_ref=T_REF,
                      max_picard=15, tol=1e-4):
    """Backward-Euler FEM with a Picard loop per step.

    Returns ``(results, total_picard_iterations)``.
    """
    n_dt, dt = _n_steps()
    T = initial_field(ic_vals)

    times = sorted(times)
    results: dict[float, np.ndarray] = {}
    t_current = 0.0
    next_idx = 0
    if times and times[0] <= 1e-9:
        results[times[0]] = T.copy()
        next_idx = 1

    total_iters = 0
    for _ in range(n_dt):
        if next_idx >= len(times):
            break
        rhs = M @ T
        rhs[0] = HEATER

        T_guess = T.copy()
        for _ in range(max_picard):
            A = M + dt * assemble_K_nonlinear(T_guess, alpha0, beta, T_ref)
            A[0, :] = 0.0
            A[0, 0] = 1.0
            T_new = np.linalg.solve(A, rhs)
            total_iters += 1
            converged = np.max(np.abs(T_new - T_guess)) < tol
            T_guess = T_new
            if converged:
                break
        T = T_guess

        t_current += dt
        while next_idx < len(times) and t_current >= times[next_idx] - 1e-6:
            results[times[next_idx]] = T.copy()
            next_idx += 1

    return results, total_iters


def solve_fem(ic_vals, alpha, times, mode: str = "linear", beta: float | None = None):
    """Convenience wrapper returning ``{t: field}`` for either mode."""
    if mode == "linear":
        return run_fem(ic_vals, alpha, times)
    if mode == "nonlinear":
        res, _ = run_fem_nonlinear(ic_vals, alpha, BETA_NL if beta is None else beta, times)
        return res
    raise ValueError(f"unknown mode '{mode}'")
