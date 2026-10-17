"""Scenario sampling and FEM-labelled training examples."""
from __future__ import annotations

import numpy as np
import torch

from .config import HEATER, L, MAX_TIME, ALPHAS, NUM_NODES, SENSOR_X, SENSORS, X_NODES
from .fem import run_fem, run_fem_nonlinear


def rand_ic(rng: np.random.Generator) -> np.ndarray:
    """Random starting temperature profile: flat base + 1-3 Gaussian bumps, heater at x=0."""
    base = rng.uniform(15.0, 30.0)
    temps = np.full(SENSORS, base)
    for _ in range(rng.integers(1, 4)):
        center = rng.uniform(0.1, L)
        width = rng.uniform(0.05, 0.3)
        height = rng.uniform(-10, 20)
        temps += height * np.exp(-((SENSOR_X - center) ** 2) / (2 * width ** 2))
    temps[0] = HEATER
    return temps


def rand_mat(rng: np.random.Generator) -> float:
    return float(rng.uniform(*ALPHAS))


def make_list(n: int, rng: np.random.Generator):
    return [(rand_ic(rng), rand_mat(rng)) for _ in range(n)]


def extract(ic_vals, alpha, rng, n_samples: int = 80, beta: float | None = None):
    """Run the FEM once and sample ``n_samples`` random (x, t) points from it.

    If ``beta`` is given the nonlinear alpha(T) solver is used.
    Each example is ``(ic, alpha, x, t, T_true)``.
    """
    times = sorted(rng.uniform(0, MAX_TIME, size=12).tolist() + [0.0])
    if beta is None:
        fem_result = run_fem(ic_vals, alpha, times)
    else:
        fem_result, _ = run_fem_nonlinear(ic_vals, alpha, beta, times)

    egs = []
    keys = list(fem_result.keys())
    for _ in range(n_samples):
        t_val = rng.choice(keys)
        x_idx = rng.integers(0, NUM_NODES)
        egs.append((ic_vals, alpha, X_NODES[x_idx], t_val, fem_result[t_val][x_idx]))
    return egs


def convert(egs):
    """List of examples -> tensors (ic, alpha, x, t, T)."""
    ic = torch.tensor(np.array([e[0] for e in egs]), dtype=torch.float32)
    alpha_t = torch.tensor([e[1] for e in egs], dtype=torch.float32)
    x_t = torch.tensor([e[2] for e in egs], dtype=torch.float32)
    t_t = torch.tensor([e[3] for e in egs], dtype=torch.float32)
    T_t = torch.tensor([e[4] for e in egs], dtype=torch.float32)
    return ic, alpha_t, x_t, t_t, T_t
