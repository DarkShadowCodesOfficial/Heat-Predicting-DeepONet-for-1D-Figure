"""Fast operator inference on (x, t) grids for arbitrary user-supplied scenarios."""
from __future__ import annotations

import numpy as np
import torch

from .config import X_NODES
from .pipeline import TrainedBundle


class OperatorService:
    """Wraps the trained operators.

    mode "linear"    -> ensemble of DeepONets for constant alpha (mean + std)
    mode "nonlinear" -> single DeepONet trained on alpha(T) FEM data (no std)
    """

    def __init__(self, bundle: TrainedBundle):
        self.bundle = bundle
        self.linear = bundle.linear_models
        self.nonlinear = bundle.nonlinear_model
        self.meta = bundle.meta

    def models(self, mode: str):
        if mode == "linear":
            return self.linear
        if mode == "nonlinear":
            return [self.nonlinear]
        raise ValueError(f"unknown mode '{mode}'")

    def predict_field(self, mode: str, ic, alpha: float, times):
        """Return ``(mean, std)`` arrays of shape (n_t, n_x) on the FEM node grid."""
        times = np.asarray(times, dtype=np.float32)
        n_t, n_x = len(times), len(X_NODES)
        x = torch.from_numpy(np.tile(X_NODES.astype(np.float32), n_t))
        t = torch.from_numpy(np.repeat(times, n_x))
        ic_t = torch.tensor(np.asarray(ic, dtype=np.float32))
        preds = torch.stack([m.forward_shared(ic_t, alpha, x, t) for m in self.models(mode)])
        mean = preds.mean(dim=0).numpy().reshape(n_t, n_x)
        std = (preds.std(dim=0).numpy().reshape(n_t, n_x)
               if preds.shape[0] > 1 else np.zeros_like(mean))
        return mean, std

    def predict_snapshot(self, mode: str, ic, alpha: float, t: float):
        mean, std = self.predict_field(mode, ic, alpha, [t])
        return mean[0], std[0]

    @staticmethod
    def has_uncertainty(mode: str) -> bool:
        return mode == "linear"
