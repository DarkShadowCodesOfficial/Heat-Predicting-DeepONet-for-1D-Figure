"""Closed-form solution for a *uniform* initial condition (notebook section 8)."""
from __future__ import annotations

import numpy as np

from .config import HEATER, L


def analytical_solution(x_grid, t_grid, alpha, L=L, heater=HEATER, T0=20.0, n_terms=200):
    """T(x,t) = heater - 4/pi (heater-T0) sum_n sin(l_n x) exp(-alpha t l_n^2)/(2n+1),
    with l_n = (2n+1) pi / (2L). Fixed T=heater at x=0, insulated at x=L.

    ``x_grid`` and ``t_grid`` are arrays of identical shape (n_t, n_x).
    """
    n = np.arange(n_terms)
    coeff = 2 * n + 1
    lam = coeff * np.pi / (2 * L)
    sin_term = np.sin(lam[:, None, None] * x_grid[None, :, :])
    exp_term = np.exp(-alpha * t_grid[None, :, :] * lam[:, None, None] ** 2)
    series = (sin_term * exp_term) / coeff[:, None, None]
    return heater - (4.0 / np.pi) * (heater - T0) * series.sum(axis=0)


def analytical_field(x_nodes, times, alpha, T0):
    """(n_t, n_x) analytical field on a rectilinear (t, x) grid."""
    X, Tm = np.meshgrid(x_nodes, times)
    return analytical_solution(X, Tm, alpha, T0=T0)


def uniform_ic_value(ic, tol=1e-6):
    """Return T0 if the IC is uniform apart from the heater point, else None."""
    rest = np.asarray(ic, dtype=float)[1:]
    if np.max(rest) - np.min(rest) < tol:
        return float(rest.mean())
    return None
