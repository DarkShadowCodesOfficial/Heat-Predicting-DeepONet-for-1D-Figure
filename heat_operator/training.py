"""Training loops (data loss + PDE-residual loss), notebook sections 6-7 and 9."""
from __future__ import annotations

import numpy as np
import torch

from .data import convert
from .model import DeepONet
from .physics import physics_loss


def train1(model, train_egs, pool, epochs, *, w_data=1.0, w_phy=0.02, lr=2e-3,
           beta=None, callback=None, log_every=500):
    """Train one DeepONet.

    ``pool``    list of (ic, alpha) scenarios the physics residual is sampled from.
    ``beta``    None -> linear heat equation, float -> nonlinear alpha(T) residual.
    ``callback(epoch, epochs, data_loss, physics_loss)`` is called every epoch.
    """
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-5)
    ic_t, alpha_t, x_t, t_t, T_t = convert(train_egs)

    for epoch in range(epochs):
        opt.zero_grad()
        pred = model(ic_t, alpha_t, x_t, t_t)
        data_loss = torch.mean((pred - T_t) ** 2)

        ic_np, alpha_val = pool[np.random.randint(len(pool))]
        phys_loss = physics_loss(model, ic_np, alpha_val, beta=beta)

        loss = w_data * data_loss + w_phy * phys_loss
        loss.backward()
        opt.step()
        sched.step()

        if callback is not None:
            callback(epoch, epochs, data_loss.item(), phys_loss.item())
    return model


def train_ensemble(train_egs, pool, epochs, *, ens_n=3, existing=None,
                   w_data=1.0, w_phy=0.02, callback=None):
    """Train (or fine-tune, if ``existing``) an ensemble of DeepONets."""
    models = []
    for i in range(ens_n):
        if existing is not None:
            model = existing[i]
        else:
            torch.manual_seed(100 + i)
            model = DeepONet()
        cb = None if callback is None else (lambda *a, _i=i: callback(_i, *a))
        models.append(train1(model, train_egs, pool, epochs, w_data=w_data,
                             w_phy=w_phy, callback=cb))
    return models
