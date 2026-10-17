# Heat-equation neural operator: training pipeline + interactive demo

A physics-informed **DeepONet ensemble** that predicts 1-D heat conduction for *any* starting
temperature profile and diffusivity, trained with FEM data and uncertainty-guided active learning.
This repo turns the original notebook into a package, plus a browser demo where you sketch an
initial condition and compare the operator against FEM (and the closed-form solution when one exists).

> **Note on naming.** The model in the notebook is a DeepONet (branch/trunk networks), not a
> Fourier Neural Operator. It is kept exactly as written; the code and UI call it "the operator".

## Quick start

```bash
git clone <your-repo-url> && cd heat-operator-demo
python -m venv .venv && source .venv/bin/activate     # optional
pip install -e .                                      # or: pip install -r requirements.txt
python -m demo                                        # opens http://127.0.0.1:8000
```

The first launch **trains the operator** (progress bar + live log in the browser). Weights are cached
in `checkpoints/<preset>/`, so later launches load in a second or two.

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

## What differs from the notebook

Results are meant to be the same; these are the intentional changes:

* FEM solvers stop at the last requested time instead of always running to 3600 s, and the
  Picard stiffness assembly is vectorised (a test checks it matches the looped version to 1e-12).
* Inference evaluates the branch network once per scenario (`DeepONet.forward_shared`) rather than once per query point. A test checks it equals `forward`.
* The standalone "7.1 initial ensemble" is not trained separately; the active-learning round 0 is that same initial ensemble.
* Randomness is seeded per pipeline stage, so individual numbers will not match your notebook run digit for digit.

## Honest expectations

* **Accuracy is modest, same as the notebook.** The notebook reports ~1.9 °C held-out MAE for the linear operator, about 4 °C in
  the first 5 minutes (≈6 °C near the heater), and ~14 °C for the α(T) operator, which the notebook itself labels a proof of concept.
  The `quick` preset gave ~2.5 °C on its held-out set and ~6 °C for the α(T) operator. Errors concentrate in the early transient near the heater;
  the demo is designed to make that visible. The ±2σ band is an ensemble disagreement signal, not a calibrated interval (the UI reports its empirical coverage).
* **Speed-ups are relative to this FEM.** It is a dense-LU solver on a 101-node 1-D mesh. On one CPU core the operator was
  roughly 5-6× faster for the full space-time field, ~25-100× for a single snapshot, and ~1000×+ against the Picard α(T) FEM. A banded solver would narrow the linear gap.
* **Out-of-distribution inputs** (jagged profiles, α outside the trained range, a heater other than 100 °C) are not supported or are expected to be inaccurate. The slider is limited to the trained α range.
* The α(T) operator is trained for one fixed β. Making β a branch-network input is the natural extension noted in the notebook.

## Tests

```bash
pip install -e ".[dev]"
pytest
```

Tests cover FEM correctness (vs the notebook's loops, vs the analytical solution, β=0 reduces to linear), model/physics
losses, a tiny end-to-end training run, checkpoint round-trip, and the HTTP API. CI config is in `.github/workflows/ci.yml`.

## API

`GET /api/status`, `GET /api/config`, `GET /api/random_ic`, `POST /api/simulate`, `POST /api/snapshot`
(request: `{ic: [21 floats], alpha, mode: "linear"|"nonlinear", t, reference: "fem"|"analytical"}`). Interactive docs at `/docs`.

## Run it from GitHub (Codespaces)

GitHub Pages cannot host this (it needs a Python server). Instead, on the repo page click
**Code → Codespaces → Create codespace on main**. The included `.devcontainer/` installs
the dependencies and starts the demo; when the port-8000 popup appears, open it in the browser.
The first start trains the operator (a few minutes on a Codespaces machine).

## Publishing to GitHub

```bash
git init && git add . && git commit -m "Initial commit"
git branch -M main
git remote add origin https://github.com/<you>/heat-operator-demo.git
git push -u origin main
```
