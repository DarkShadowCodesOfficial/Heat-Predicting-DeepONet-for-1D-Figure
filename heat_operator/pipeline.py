"""End-to-end training pipeline: data generation -> active learning -> nonlinear operator.

This is the notebook (sections 1-9) as a reusable function with progress reporting.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import torch

from .active_learning import FEM_UNITS, active_learn, evaluate, predict
from .config import (BETA_NL, MAX_TIME, NUM_NODES, TrainConfig, X_NODES)
from .data import extract, make_list, rand_ic, rand_mat
from .fem import run_fem, run_fem_nonlinear
from .model import DeepONet
from .progress import Progress
from .training import train1

# progress weight of one nonlinear-FEM scenario relative to one training epoch
NL_FEM_UNITS = 120.0


@dataclass
class TrainedBundle:
    linear_models: list
    nonlinear_model: DeepONet
    meta: dict = field(default_factory=dict)


def estimate_units(cfg: TrainConfig) -> float:
    n_strat = 2 if cfg.run_baseline else 1
    epochs = n_strat * (cfg.ens_n * cfg.epochs + cfg.rounds * cfg.ens_n * cfg.epochs_round)
    fem = FEM_UNITS * (cfg.inits + cfg.tests + n_strat * cfg.rounds * cfg.k)
    nl = NL_FEM_UNITS * cfg.nl_scenarios + cfg.nl_epochs
    return epochs + fem + nl


def _epoch_logger(progress: Progress, label: str):
    """Returns a factory ``tag -> callback`` that advances progress and logs sparsely."""
    def factory(tag):
        def cb(model_idx, epoch, epochs, data_loss, phys_loss):
            progress.advance(1)
            if epoch % 500 == 0 or epoch == epochs - 1 and epochs < 500:
                progress.add_log(
                    f"  {label}/{tag} model {model_idx + 1} epoch {epoch}/{epochs}: "
                    f"data {data_loss:.4f}  physics {phys_loss:.4f}")
        return cb
    return factory


def run_pipeline(cfg: TrainConfig, progress: Progress | None = None) -> TrainedBundle:
    progress = progress or Progress()
    progress.set_total(estimate_units(cfg))
    t_start = time.time()

    np.random.seed(0)
    torch.manual_seed(0)
    rng = np.random.default_rng(cfg.seed)

    # ---------------- scenario pools + FEM-labelled data ----------------
    init_pool = make_list(cfg.inits, rng)
    choice_pool = make_list(cfg.choices, rng)          # not simulated yet
    test_pool = make_list(cfg.tests, rng)

    progress.set_stage("FEM data", f"simulating {cfg.inits} initial scenarios")
    t_fem = time.time()
    train_egs = []
    for ic, alpha in init_pool:
        train_egs.extend(extract(ic, alpha, rng, cfg.n_samples))
        progress.advance(FEM_UNITS)

    progress.set_stage("FEM data", f"simulating {cfg.tests} held-out test scenarios")
    femcache = {}
    for i, (ic, alpha) in enumerate(test_pool):
        femcache[i] = run_fem(ic, alpha, list(np.linspace(0, MAX_TIME, 8)))
        progress.advance(FEM_UNITS)
    fem_data_time = time.time() - t_fem

    common = dict(init_pool=init_pool, choice_pool=choice_pool, test_pool=test_pool,
                  train_egs=train_egs, femcache=femcache, rng=rng, progress=progress)

    # ---------------- uncertainty-guided active learning ----------------
    t_al = time.time()
    al_models, al_hist = active_learn("uncertainty", 0, cfg, **common,
                                      callback_factory=_epoch_logger(progress, "AL"))
    al_time = time.time() - t_al

    bl_hist = None
    if cfg.run_baseline:
        _, bl_hist = active_learn("random", 1, cfg, **common,
                                  callback_factory=_epoch_logger(progress, "random"))

    mean_err, max_err = evaluate(al_models, test_pool, femcache)

    # ---------------- nonlinear alpha(T) operator (section 9) ----------------
    progress.set_stage("nonlinear FEM data",
                       f"Picard-FEM on {cfg.nl_scenarios} scenarios (beta={cfg.beta})")
    nl_rng = np.random.default_rng(21)
    nl_pool = [(rand_ic(nl_rng), rand_mat(nl_rng)) for _ in range(cfg.nl_scenarios)]
    nl_egs = []
    t_nl = time.time()
    for ic, a in nl_pool:
        nl_egs.extend(extract(ic, a, nl_rng, cfg.n_samples, beta=cfg.beta))
        progress.advance(NL_FEM_UNITS)

    progress.set_stage("nonlinear operator", f"training for {cfg.nl_epochs} epochs")
    torch.manual_seed(300)
    nl_model = DeepONet()

    def nl_cb(epoch, epochs, dl, pl):
        progress.advance(1)
        if epoch % 500 == 0:
            progress.add_log(f"  nonlinear epoch {epoch}/{epochs}: data {dl:.4f}  physics {pl:.4f}")

    nl_model = train1(nl_model, nl_egs, nl_pool, cfg.nl_epochs, w_data=cfg.w_data,
                      w_phy=cfg.w_phy, beta=cfg.beta, callback=nl_cb)
    nl_time = time.time() - t_nl

    nl_mae = _nonlinear_check(nl_model, cfg.beta)
    total = time.time() - t_start
    progress.set_stage("done", f"trained in {total:.0f}s")

    meta = {
        "config": cfg.to_dict(),
        "al_history": al_hist,
        "baseline_history": bl_hist,
        "final_test_mean_err": mean_err,
        "final_test_max_err": max_err,
        "final_train_examples": al_hist[-1]["size"],
        "initial_train_examples": len(train_egs),
        "nonlinear_train_examples": len(nl_egs),
        "nonlinear_test_mae": nl_mae,
        "timings_s": {"total": total, "fem_data": fem_data_time,
                      "active_learning": al_time, "nonlinear": nl_time},
        "torch_version": torch.__version__,
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    return TrainedBundle(al_models, nl_model, meta)


def _nonlinear_check(nl_model, beta) -> float:
    """MAE vs nonlinear FEM on a held-out scenario (notebook section 9)."""
    rng = np.random.default_rng(99)
    ic, alpha = rand_ic(rng), rand_mat(rng)
    times = [60, 150, 300, 600]
    fem, _ = run_fem_nonlinear(ic, alpha, beta, times)
    errs = []
    for t in times:
        n = NUM_NODES
        with torch.no_grad():
            pred = nl_model(torch.tensor(np.tile(ic, (n, 1)), dtype=torch.float32),
                            torch.full((n,), float(alpha)),
                            torch.tensor(X_NODES, dtype=torch.float32),
                            torch.full((n,), float(t))).numpy()
        errs.append(np.mean(np.abs(pred - fem[t])))
    return float(np.mean(errs))
