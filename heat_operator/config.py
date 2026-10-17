"""Physical constants and training presets.

Everything here mirrors the module-level constants of the original notebook
(`notebooks/self_improving_heat_operator_nonlinear_alpha.ipynb`).
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np

# --------------------------------------------------------------------------- #
# Physics / discretisation (identical to the notebook)
# --------------------------------------------------------------------------- #
L = 1.0                     # rod length
MAX_TIME = 3600.0           # max simulated time (s)
HEATER = 100.0              # Dirichlet temperature at x = 0 (degC)
ALPHAS = (5e-4, 2.5e-3)     # range of diffusivities the operator is trained on
SENSORS = 21                # number of points describing the initial condition

NUM_NODES = 101             # FEM nodes
TIME_STEP = 0.5             # FEM time step (s)
T_REF = 20.0                # reference temperature for alpha(T)
BETA_NL = -0.003            # "gold-like" temperature sensitivity for alpha(T)

SENSOR_X = np.linspace(0.0, L, SENSORS)
X_NODES = np.linspace(0.0, L, NUM_NODES)
H = X_NODES[1] - X_NODES[0]


# --------------------------------------------------------------------------- #
# Training configuration
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class TrainConfig:
    """All hyper-parameters of the training pipeline.

    The ``notebook`` preset reproduces the notebook's numbers exactly.
    """
    # scenario pools
    inits: int = 60             # initial training scenarios
    choices: int = 60           # candidate pool for active learning
    tests: int = 15             # held-out test scenarios
    n_samples: int = 80         # (x, t) samples drawn per FEM run
    # ensemble / loss
    ens_n: int = 3
    w_data: float = 1.0
    w_phy: float = 0.02
    # training lengths
    epochs: int = 1500          # initial training
    # active learning
    rounds: int = 4
    k: int = 6                  # scenarios acquired per round
    epochs_round: int = 250     # fine-tuning epochs per round
    run_baseline: bool = True   # also run the random-selection baseline
    # nonlinear alpha(T) operator
    nl_scenarios: int = 20
    nl_epochs: int = 1500
    beta: float = BETA_NL
    # reproducibility
    seed: int = 42

    def to_dict(self) -> dict:
        return asdict(self)


PRESETS: dict[str, TrainConfig] = {
    # Exactly the notebook's hyper-parameters.
    "notebook": TrainConfig(),
    # Same pipeline, smaller budget. A good default for a demo.
    "quick": TrainConfig(
        inits=40, choices=30, tests=8, epochs=800, rounds=3, k=5,
        epochs_round=150, run_baseline=False, nl_scenarios=12, nl_epochs=800,
    ),
    # Smoke-test size, used by the unit tests / CI. The operator is NOT accurate.
    "tiny": TrainConfig(
        inits=4, choices=4, tests=2, n_samples=20, ens_n=2, epochs=20,
        rounds=1, k=2, epochs_round=10, run_baseline=True,
        nl_scenarios=2, nl_epochs=20,
    ),
}


def get_preset(name: str) -> TrainConfig:
    if name not in PRESETS:
        raise KeyError(f"Unknown preset '{name}'. Choose from {sorted(PRESETS)}")
    return PRESETS[name]
