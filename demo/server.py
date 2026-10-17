"""FastAPI backend for the interactive demo.

On start-up the server either loads a saved checkpoint or trains the operators
in a background thread (progress is exposed at ``/api/status`` and shown by the
UI). Once ready, ``/api/simulate`` and ``/api/snapshot`` compare the operator
with the FEM (and the analytical solution, when one exists) for any initial
condition, diffusivity and time.
"""
from __future__ import annotations

import logging
import os
import threading
import traceback
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

from heat_operator.analytical import analytical_field, uniform_ic_value
from heat_operator.analysis import (N_X_SEGMENTS, TIME_BINS, error_metrics,
                                    interval_errors, locate_max, segment_masks, timed)
from heat_operator.config import (ALPHAS, BETA_NL, HEATER, L, MAX_TIME, NUM_NODES,
                                  SENSOR_X, SENSORS, X_NODES, get_preset)
from heat_operator.data import rand_ic, rand_mat
from heat_operator.fem import solve_fem
from heat_operator.inference import OperatorService
from heat_operator.persistence import load_bundle, save_bundle
from heat_operator.pipeline import run_pipeline
from heat_operator.progress import Progress

log = logging.getLogger("heat_operator.demo")
STATIC = Path(__file__).parent / "static"


# --------------------------------------------------------------------------- #
# Request schema
# --------------------------------------------------------------------------- #
class SimRequest(BaseModel):
    ic: list[float] = Field(..., description=f"{SENSORS} initial temperatures at x = 0, 0.05, ..., 1")
    alpha: float = Field(..., description="(base) thermal diffusivity alpha0")
    mode: Literal["linear", "nonlinear"] = "linear"
    t: float = Field(600.0, ge=0.0, le=MAX_TIME)
    reference: Literal["fem", "analytical"] = "fem"
    n_times: int = Field(121, ge=11, le=241)

    @field_validator("ic")
    @classmethod
    def _check_ic(cls, v):
        if len(v) != SENSORS:
            raise ValueError(f"ic must have exactly {SENSORS} values")
        if not all(np.isfinite(v)) or min(v) < -100 or max(v) > 300:
            raise ValueError("ic values must be finite and within [-100, 300] degC")
        return v

    @field_validator("alpha")
    @classmethod
    def _check_alpha(cls, v):
        lo, hi = ALPHAS
        if not (lo * 0.999 <= v <= hi * 1.001):
            raise ValueError(f"alpha must lie in the trained range [{lo}, {hi}]")
        return v


def _r(a, nd=3):
    return np.round(np.asarray(a, dtype=float), nd).tolist()


# --------------------------------------------------------------------------- #
# Application state
# --------------------------------------------------------------------------- #
class AppState:
    def __init__(self, preset: str, ckpt_root: str, retrain: bool):
        self.preset = preset
        self.cfg = get_preset(preset)
        self.ckpt_dir = Path(ckpt_root) / preset
        self.retrain = retrain
        self.state = "loading"
        self.error: str | None = None
        self.from_checkpoint = False
        self.progress = Progress()
        self.service: OperatorService | None = None
        self.lock = threading.Lock()          # serialises compute so timings are clean

    def boot(self):
        try:
            bundle = None if self.retrain else load_bundle(self.ckpt_dir, self.cfg)
            if bundle is not None:
                self.progress.set_stage("loading", f"loaded checkpoint from {self.ckpt_dir}")
                self.from_checkpoint = True
            else:
                self.state = "training"
                bundle = run_pipeline(self.cfg, self.progress)
                save_bundle(bundle, self.ckpt_dir)
                self.progress.add_log(f"saved checkpoint to {self.ckpt_dir}")
            self.service = OperatorService(bundle)
            self._warmup()
            self.state = "ready"
        except Exception as exc:                                  # pragma: no cover
            self.state = "error"
            self.error = f"{type(exc).__name__}: {exc}"
            log.error("boot failed\n%s", traceback.format_exc())

    def _warmup(self):
        ic = np.full(SENSORS, 20.0)
        ic[0] = HEATER
        for mode in ("linear", "nonlinear"):
            self.service.predict_snapshot(mode, ic, 1e-3, 100.0)


# --------------------------------------------------------------------------- #
# Core computations
# --------------------------------------------------------------------------- #
def _prep(req: SimRequest):
    ic = np.array(req.ic, dtype=float)
    ic[0] = HEATER                      # the heater end is fixed by the physics
    T0 = uniform_ic_value(ic) if req.mode == "linear" else None
    return ic, T0


def _fem_grid(ic, alpha, mode, times):
    tl = [float(t) for t in times]
    res = solve_fem(ic, alpha, tl, mode)
    return np.array([res[t] for t in tl])


def _snapshot(app: AppState, req: SimRequest, ic, T0, want_timing: bool, fem_row=None):
    svc, t = app.service, float(req.t)

    if fem_row is None or want_timing:
        res, fem_s = timed(lambda: solve_fem(ic, req.alpha, [t], req.mode), 1)
        fem_row = res[t]
    else:
        fem_s = None
    svc.predict_snapshot(req.mode, ic, req.alpha, t)            # warm
    (op, std), op_s = timed(lambda: svc.predict_snapshot(req.mode, ic, req.alpha, t),
                            20 if want_timing else 1)

    ana = analytical_field(X_NODES, [t], req.alpha, T0)[0] if T0 is not None else None
    use_ana = req.reference == "analytical" and ana is not None
    ref = ana if use_ana else fem_row

    out = {
        "t": t,
        "fem": _r(fem_row), "op": _r(op), "std": _r(std, 4),
        "analytical": None if ana is None else _r(ana),
        "err": _r(op - ref, 4),
        "reference": "analytical" if use_ana else "fem",
        "metrics": error_metrics(op, ref, std if svc.has_uncertainty(req.mode) else None),
        "by_x_segment": [float(np.mean(np.abs(op - ref)[m]))
                         for m in segment_masks(X_NODES)[1]],
    }
    if want_timing:
        out["timing"] = {
            "fem_s": fem_s, "op_s": op_s,
            "speedup": (fem_s / op_s) if (op_s > 0 and t > 0) else None,
            "fem_steps": int(round(t / (MAX_TIME / int(MAX_TIME / 0.5)))),
        }
    return out


def create_app(preset: str = "quick", ckpt_root: str = "checkpoints",
               retrain: bool = False) -> FastAPI:
    state = AppState(preset, ckpt_root, retrain)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        threading.Thread(target=state.boot, daemon=True, name="trainer").start()
        yield

    app = FastAPI(title="Heat operator demo", lifespan=lifespan)
    app.state.demo = state
    app.mount("/static", StaticFiles(directory=STATIC), name="static")

    def require_ready():
        if state.state != "ready":
            raise HTTPException(503, f"operator not ready (state={state.state})")

    # ------------------------------------------------------------------ pages
    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(STATIC / "index.html")

    # -------------------------------------------------------------- meta APIs
    @app.get("/api/status")
    def status():
        return {"state": state.state, "preset": state.preset, "error": state.error,
                "from_checkpoint": state.from_checkpoint,
                "progress": state.progress.snapshot()}

    @app.get("/api/config")
    def config():
        out = {
            "sensors": SENSORS, "sensor_x": SENSOR_X.tolist(), "x_nodes": X_NODES.tolist(),
            "length": L, "max_time": MAX_TIME, "heater": HEATER,
            "alpha_range": list(ALPHAS), "beta": state.cfg.beta,
            "time_bins": [list(b) for b in TIME_BINS], "n_x_segments": N_X_SEGMENTS,
            "preset": state.preset, "ready": state.state == "ready", "training": None,
        }
        if state.service is not None:
            out["training"] = state.service.meta
        return out

    @app.get("/api/random_ic")
    def random_ic(seed: int | None = None):
        rng = np.random.default_rng(seed)
        return {"ic": _r(rand_ic(rng), 2), "alpha": rand_mat(rng)}

    # ------------------------------------------------------------ main compute
    @app.post("/api/simulate")
    def simulate(req: SimRequest):
        require_ready()
        with state.lock:
            svc = state.service
            ic, T0 = _prep(req)
            times = np.linspace(0.0, MAX_TIME, req.n_times)

            fem_grid, fem_full_s = timed(lambda: _fem_grid(ic, req.alpha, req.mode, times), 1)
            svc.predict_field(req.mode, ic, req.alpha, times)                 # warm
            (op_grid, std_grid), op_full_s = timed(
                lambda: svc.predict_field(req.mode, ic, req.alpha, times), 5)

            ana_grid = analytical_field(X_NODES, times, req.alpha, T0) if T0 is not None else None
            use_ana = req.reference == "analytical" and ana_grid is not None
            ref_grid = ana_grid if use_ana else fem_grid
            err_grid = op_grid - ref_grid

            has_std = svc.has_uncertainty(req.mode)
            # reuse the FEM row when the requested time lies on the grid
            k = int(np.argmin(np.abs(times - req.t)))
            fem_row = fem_grid[k] if abs(times[k] - req.t) < 1e-6 else None
            snap = _snapshot(state, req, ic, T0, want_timing=False, fem_row=fem_row)

            return {
                "x": X_NODES.tolist(), "times": times.tolist(),
                "mode": req.mode, "alpha": req.alpha, "beta": state.cfg.beta,
                "reference": "analytical" if use_ana else "fem",
                "analytical_available": ana_grid is not None,
                "uniform_T0": T0,
                "has_uncertainty": has_std,
                "fem_grid": _r(fem_grid), "op_grid": _r(op_grid),
                "std_grid": _r(std_grid, 4), "err_grid": _r(err_grid, 4),
                "analytical_grid": None if ana_grid is None else _r(ana_grid),
                "metrics": error_metrics(op_grid, ref_grid, std_grid if has_std else None),
                "max_error_at": locate_max(err_grid, times, X_NODES),
                "intervals": interval_errors(err_grid, times, X_NODES),
                "fem_vs_analytical": (
                    {"mae": float(np.mean(np.abs(fem_grid - ana_grid))),
                     "max": float(np.max(np.abs(fem_grid - ana_grid)))}
                    if ana_grid is not None else None),
                "timing": {
                    "fem_full_s": fem_full_s, "op_full_s": op_full_s,
                    "speedup_full": fem_full_s / op_full_s if op_full_s > 0 else None,
                    "grid_points": int(req.n_times * NUM_NODES),
                },
                "snapshot": snap,
            }

    @app.post("/api/snapshot")
    def snapshot(req: SimRequest):
        require_ready()
        with state.lock:
            ic, T0 = _prep(req)
            return {"snapshot": _snapshot(state, req, ic, T0, want_timing=True)}

    return app


app = create_app(os.environ.get("HEAT_PRESET", "quick"),
                 os.environ.get("HEAT_CKPT_DIR", "checkpoints"),
                 os.environ.get("HEAT_RETRAIN", "0") == "1")
