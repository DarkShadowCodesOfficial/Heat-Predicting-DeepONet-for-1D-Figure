"""PDE-residual losses (physics-informed term)."""
from __future__ import annotations

import numpy as np
import torch

from .config import L, MAX_TIME, T_REF


def physics_loss(model, ic_np, alpha_val: float, beta: float | None = None,
                 n_phys: int = 64, T_ref: float = T_REF) -> torch.Tensor:
    """Mean squared PDE residual at random collocation points for one scenario.

    * ``beta is None``  : dT/dt - alpha d2T/dx2 = 0
    * ``beta`` given    : dT/dt - alpha(T) d2T/dx2 - alpha0*beta (dT/dx)^2 = 0
    """
    ic_phys = torch.tensor(np.tile(ic_np, (n_phys, 1)), dtype=torch.float32)
    alpha_phys = torch.full((n_phys,), float(alpha_val), dtype=torch.float32)
    x_phys = (torch.rand(n_phys) * L).requires_grad_(True)
    t_phys = (torch.rand(n_phys) * MAX_TIME).requires_grad_(True)

    T_phys = model(ic_phys, alpha_phys, x_phys, t_phys)
    ones = torch.ones_like(T_phys)
    dT_dx = torch.autograd.grad(T_phys, x_phys, ones, create_graph=True)[0]
    d2T_dx2 = torch.autograd.grad(dT_dx, x_phys, torch.ones_like(dT_dx), create_graph=True)[0]
    dT_dt = torch.autograd.grad(T_phys, t_phys, ones, create_graph=True)[0]

    if beta is None:
        residual = dT_dt - alpha_val * d2T_dx2
    else:
        alpha_T = alpha_phys * (1.0 + beta * (T_phys - T_ref))
        dalpha_dT = alpha_phys * beta
        residual = dT_dt - alpha_T * d2T_dx2 - dalpha_dT * (dT_dx ** 2)
    return torch.mean(residual ** 2)
