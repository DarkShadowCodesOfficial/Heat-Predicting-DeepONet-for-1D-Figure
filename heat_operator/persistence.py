"""Save / load trained operators so the demo only trains once."""
from __future__ import annotations

import json
from pathlib import Path

import torch

from .config import TrainConfig
from .model import DeepONet
from .pipeline import TrainedBundle

WEIGHTS = "operators.pt"
META = "meta.json"


def save_bundle(bundle: TrainedBundle, ckpt_dir) -> Path:
    d = Path(ckpt_dir)
    d.mkdir(parents=True, exist_ok=True)
    torch.save({
        "linear": [m.state_dict() for m in bundle.linear_models],
        "nonlinear": bundle.nonlinear_model.state_dict(),
    }, d / WEIGHTS)
    (d / META).write_text(json.dumps(bundle.meta, indent=2))
    return d


def load_bundle(ckpt_dir, expected_cfg: TrainConfig | None = None) -> TrainedBundle | None:
    """Return the saved bundle, or None if missing / trained with a different config."""
    d = Path(ckpt_dir)
    if not (d / WEIGHTS).exists() or not (d / META).exists():
        return None
    meta = json.loads((d / META).read_text())
    if expected_cfg is not None and meta.get("config") != expected_cfg.to_dict():
        return None
    state = torch.load(d / WEIGHTS, map_location="cpu", weights_only=True)
    linear = []
    for sd in state["linear"]:
        m = DeepONet()
        m.load_state_dict(sd)
        m.eval()
        linear.append(m)
    nl = DeepONet()
    nl.load_state_dict(state["nonlinear"])
    nl.eval()
    return TrainedBundle(linear, nl, meta)
