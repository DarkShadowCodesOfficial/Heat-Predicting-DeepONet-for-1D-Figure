"""Ensemble prediction, uncertainty scoring and the active-learning loop (notebook section 7)."""
from __future__ import annotations

import numpy as np
import torch

from .config import L, MAX_TIME, NUM_NODES, X_NODES, TrainConfig
from .data import extract
from .training import train_ensemble


def predict(models, ic_batch, alpha_batch, x_batch, t_batch):
    """Ensemble mean and std for a batch of (ic, alpha, x, t) queries."""
    with torch.no_grad():
        preds = torch.stack([m(ic_batch, alpha_batch, x_batch, t_batch) for m in models])
    return preds.mean(dim=0), preds.std(dim=0)


def uncertain(models, ic, alpha, n_query=30):
    """Mean ensemble std over random (x, t) points: the acquisition score."""
    ic_t = torch.tensor(np.tile(ic, (n_query, 1)), dtype=torch.float32)
    alpha_t = torch.full((n_query,), float(alpha), dtype=torch.float32)
    x_t = torch.rand(n_query) * L
    t_t = torch.rand(n_query) * MAX_TIME
    _, std = predict(models, ic_t, alpha_t, x_t, t_t)
    return std.mean().item()


def evaluate(models, test_pool, femcache, rng=None):
    """Mean / max abs error on the held-out FEM test set (10 random nodes per time)."""
    rng = rng or np.random.default_rng(0)
    errors = []
    for i, (ic, alpha) in enumerate(test_pool):
        for t_val, T_field in femcache[i].items():
            n_sample = 10
            idxs = rng.choice(NUM_NODES, n_sample, replace=False)
            ic_t = torch.tensor(np.tile(ic, (n_sample, 1)), dtype=torch.float32)
            alpha_t = torch.full((n_sample,), float(alpha), dtype=torch.float32)
            x_t = torch.tensor(X_NODES[idxs], dtype=torch.float32)
            t_t = torch.full((n_sample,), float(t_val), dtype=torch.float32)
            pred, _ = predict(models, ic_t, alpha_t, x_t, t_t)
            errors.append(np.abs(pred.numpy() - T_field[idxs]))
    errors = np.concatenate(errors)
    return float(errors.mean()), float(errors.max())


def active_learn(strategy: str, seed_offset: int, cfg: TrainConfig, *, init_pool,
                 choice_pool, test_pool, train_egs, femcache, rng, progress=None,
                 callback_factory=None):
    """Initial training followed by ``cfg.rounds`` acquisition + fine-tuning rounds.

    strategy = "uncertainty" (pick scenarios with highest ensemble disagreement)
             | "random"      (baseline)
    Returns ``(models, history)``; history entries are {round, size, err}.
    """
    say = progress.add_log if progress else (lambda s: None)
    stage = (lambda s, m="": progress.set_stage(s, m)) if progress else (lambda s, m="": None)
    cb = (lambda tag: callback_factory(tag)) if callback_factory else (lambda tag: None)

    torch.manual_seed(200 + seed_offset)
    stage(f"{strategy}: initial training",
          f"{cfg.ens_n} models x {cfg.epochs} epochs on {len(train_egs)} examples")
    models = train_ensemble(train_egs, init_pool, cfg.epochs, ens_n=cfg.ens_n,
                            w_data=cfg.w_data, w_phy=cfg.w_phy, callback=cb("init"))

    current_egs = list(train_egs)
    remains = list(choice_pool)
    hist = []

    mean_err, _ = evaluate(models, test_pool, femcache)
    hist.append({"round": 0, "size": len(current_egs), "err": mean_err})
    say(f"[{strategy}] round 0  test mean error {mean_err:.3f} degC")

    for r in range(1, cfg.rounds + 1):
        k = min(cfg.k, len(remains))
        if strategy == "uncertainty":
            scores = [uncertain(models, ic, a) for ic, a in remains]
            chosen_idx = np.argsort(scores)[::-1][:k]
        else:
            chosen_idx = np.random.choice(len(remains), size=k, replace=False)

        chosen = [remains[i] for i in chosen_idx]
        drop = set(int(i) for i in chosen_idx)
        remains = [s for j, s in enumerate(remains) if j not in drop]

        stage(f"{strategy}: round {r}/{cfg.rounds}", f"running FEM on {k} acquired scenarios")
        for ic, a in chosen:
            current_egs.extend(extract(ic, a, rng, cfg.n_samples))
            if progress:
                progress.advance(FEM_UNITS)

        stage(f"{strategy}: round {r}/{cfg.rounds}",
              f"fine-tuning {cfg.ens_n} models x {cfg.epochs_round} epochs")
        models = train_ensemble(current_egs, init_pool + chosen, cfg.epochs_round,
                                ens_n=cfg.ens_n, existing=models, w_data=cfg.w_data,
                                w_phy=cfg.w_phy, callback=cb(f"round{r}"))

        mean_err, _ = evaluate(models, test_pool, femcache)
        hist.append({"round": r, "size": len(current_egs), "err": mean_err})
        say(f"[{strategy}] round {r}  size {len(current_egs)}  test mean error {mean_err:.3f} degC")

    return models, hist


# progress weight of one linear-FEM scenario relative to one training epoch
FEM_UNITS = 30.0
