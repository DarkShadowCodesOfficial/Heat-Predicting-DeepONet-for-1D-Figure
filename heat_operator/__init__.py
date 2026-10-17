"""Physics-informed DeepONet for the 1-D heat equation with FEM ground truth."""
from .config import ALPHAS, BETA_NL, HEATER, L, MAX_TIME, NUM_NODES, SENSORS, X_NODES, PRESETS, TrainConfig, get_preset
from .model import DeepONet

__all__ = ["ALPHAS", "BETA_NL", "HEATER", "L", "MAX_TIME", "NUM_NODES", "SENSORS",
           "X_NODES", "PRESETS", "TrainConfig", "get_preset", "DeepONet"]
