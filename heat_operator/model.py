"""DeepONet used in the notebook (branch net on [IC, alpha], trunk net on (x, t))."""
from __future__ import annotations

import torch
import torch.nn as nn

from .config import ALPHAS, MAX_TIME, SENSORS

P = 64  # size of the branch/trunk latent space


class DeepONet(nn.Module):
    def __init__(self, sensors: int = SENSORS, p: int = P):
        super().__init__()
        self.branch = nn.Sequential(
            nn.Linear(sensors + 1, 128), nn.Tanh(),
            nn.Linear(128, 128), nn.Tanh(),
            nn.Linear(128, p),
        )
        self.trunk = nn.Sequential(
            nn.Linear(2, 128), nn.Tanh(),
            nn.Linear(128, 128), nn.Tanh(),
            nn.Linear(128, p), nn.Tanh(),
        )
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, ic_batch, alpha_batch, x_batch, t_batch):
        alpha_norm = (alpha_batch - ALPHAS[0]) / (ALPHAS[1] - ALPHAS[0])
        branch_in = torch.cat([ic_batch, alpha_norm.unsqueeze(-1)], dim=-1)
        branch_out = self.branch(branch_in)
        t_norm = t_batch / MAX_TIME
        trunk_in = torch.stack([x_batch, t_norm], dim=-1)
        trunk_out = self.trunk(trunk_in)
        return (branch_out * trunk_out).sum(dim=-1) + self.bias

    @torch.no_grad()
    def forward_shared(self, ic, alpha: float, x_batch, t_batch):
        """Evaluate ONE scenario (``ic`` of shape (sensors,)) at many (x, t) points.

        Mathematically identical to ``forward`` with the scenario tiled, but the
        branch network runs once instead of once per query point, which is the
        whole point of the DeepONet factorisation.
        """
        alpha_norm = (float(alpha) - ALPHAS[0]) / (ALPHAS[1] - ALPHAS[0])
        branch_in = torch.cat([ic.reshape(-1), torch.tensor([alpha_norm], dtype=ic.dtype)])
        b = self.branch(branch_in)                                   # (p,)
        trunk_out = self.trunk(torch.stack([x_batch, t_batch / MAX_TIME], dim=-1))  # (N, p)
        return trunk_out @ b + self.bias
