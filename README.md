# Heat-equation neural operator: training pipeline + interactive demo

A physics-informed DeepONet ensemble that predicts 1-D heat conduction for any starting
temperature profile and diffusivity, trained with FEM data and uncertainty-guided active learning.

## Quick start

```bash
git clone <your-repo-url> && cd heat-operator-demo
python -m venv .venv && source .venv/bin/activate     # optional
pip install -e .                                      # or: pip install -r requirements.txt
python -m demo                                        # opens http://127.0.0.1:8000
```

The first launch trains the operator.

| preset | what it is | training time* |
|---|---|---|
| `quick` (default) | same pipeline, smaller budget, no random baseline | ~2 min |
| `notebook` | the notebook's exact hyper-parameters, including the random-selection baseline | several minutes |

\*single CPU core; measured for `quick`. Use `python -m demo --preset notebook`. Add `--retrain` to ignore a cache.
You can also train without the UI: `python scripts/train.py --preset quick`.

The page loads Plotly from a CDN, so the browser needs internet access.

## Using the demo

* **Initial temperature**: drag any of the 21 points up/down (click anywhere in its column). Hold
  *Shift* or tick *Smooth brush* to drag a smooth bump. x = 0 is the heater, fixed at 100 °C. Presets
  include random (drawn from the training distribution), uniform, bumps, ramp and a jagged
  out-of-distribution profile.
* **Material**: α₀ slider over the trained range, and a switch between constant α and the temperature-dependent
  α(T) = α₀(1 + β(T − 20)) operator (β = −0.003, fixed at training).
* **Time**: slider (0-3600 s) plus an animate button.
* **Tabs**
  * *Solution*: FEM, operator mean, ±2σ ensemble band, analytical curve (uniform IC only), pointwise error, MAE / RMSE / max / relative L2 / ±2σ coverage.
  * *Space-time*: FEM, operator, error and uncertainty heatmaps with a time window zoom.
  * *Error intervals*: MAE by x-segment × time window (the notebook's "first 5 minutes, five segments" analysis, generalised), plus error-vs-t and error-vs-x curves.
  * *Speed vs FEM*: measured wall-clock for one time snapshot and for the full space-time field, and a break-even estimate that includes training cost.
  * *Training*: active-learning curve and the training configuration/timings.

## Layout

```
heat_operator/        importable package (the notebook as modules)
  config.py           constants + training presets
  fem.py              linear and nonlinear (Picard) FEM
  analytical.py       closed-form solution for uniform IC
  model.py            DeepONet
  physics.py          PDE-residual losses (linear and alpha(T))
  data.py             scenario sampling, FEM-labelled examples
  training.py         train loops / ensembles
  active_learning.py  uncertainty scoring, acquisition loop
  pipeline.py         end-to-end training with progress reporting
  persistence.py      checkpoint save/load
  inference.py        fast operator evaluation
  analysis.py         error metrics, interval breakdowns, timing
demo/                 FastAPI backend (server.py) + static front-end
scripts/train.py      train from the command line
notebooks/            your original notebook, unchanged
tests/                pytest suite (FEM vs notebook, vs analytical, model, API)
```
