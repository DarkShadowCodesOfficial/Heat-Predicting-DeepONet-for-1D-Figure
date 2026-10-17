"""Error metrics, interval breakdowns and timing helpers used by the demo."""
from __future__ import annotations

import time

import numpy as np

# Time windows (s) and number of x-segments used for the interval breakdown.
# The first window / segmentation reproduces the notebook's "first 5 minutes,
# five x-segments of width 0.2" analysis.
TIME_BINS = [(0.0, 300.0), (300.0, 900.0), (900.0, 1800.0), (1800.0, 3600.0)]
N_X_SEGMENTS = 5


def error_metrics(pred, ref, std=None) -> dict:
    """Scalar error summary of ``pred - ref`` (any array shape)."""
    err = np.asarray(pred) - np.asarray(ref)
    out = {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "max": float(np.max(np.abs(err))),
        "bias": float(np.mean(err)),
        "rel_l2_pct": float(100 * np.linalg.norm(err) / max(np.linalg.norm(ref), 1e-12)),
    }
    if std is not None and np.max(std) > 0:
        out["coverage_2std_pct"] = float(100 * np.mean(np.abs(err) <= 2 * std))
        out["mean_std"] = float(np.mean(std))
    return out


def segment_masks(x_nodes, n_seg=N_X_SEGMENTS):
    edges = np.linspace(x_nodes[0], x_nodes[-1], n_seg + 1)
    masks = []
    for i in range(n_seg):
        lo, hi = edges[i], edges[i + 1]
        m = (x_nodes >= lo - 1e-12) & ((x_nodes < hi - 1e-12) if i < n_seg - 1 else (x_nodes <= hi + 1e-12))
        masks.append(m)
    return edges, masks


def _time_masks(times, bins=TIME_BINS):
    masks = []
    for i, (lo, hi) in enumerate(bins):
        m = (times <= hi + 1e-9) & ((times >= lo - 1e-9) if i == 0 else (times > lo + 1e-9))
        masks.append(m)
    return masks


def interval_errors(err, times, x_nodes) -> dict:
    """MAE of ``err`` (n_t, n_x) over time windows x spatial segments."""
    times = np.asarray(times)
    x_nodes = np.asarray(x_nodes)
    a = np.abs(err)
    edges, xm = segment_masks(x_nodes)
    tm = _time_masks(times)
    matrix = np.full((len(xm), len(tm)), np.nan)
    for i, mx in enumerate(xm):
        for j, mt in enumerate(tm):
            if mx.any() and mt.any():
                matrix[i, j] = a[np.ix_(mt, mx)].mean()
    by_x = [float(a[:, mx].mean()) for mx in xm]
    by_t = [float(a[mt, :].mean()) if mt.any() else None for mt in tm]
    return {
        "x_edges": edges.tolist(),
        "time_bins": [list(b) for b in TIME_BINS],
        "matrix": [[None if np.isnan(v) else float(v) for v in row] for row in matrix],
        "by_x_segment": by_x,
        "by_time_bin": by_t,
        "mae_curve_vs_time": a.mean(axis=1).tolist(),
        "mae_curve_vs_x": a.mean(axis=0).tolist(),
    }


def locate_max(err, times, x_nodes) -> dict:
    i, j = np.unravel_index(np.argmax(np.abs(err)), err.shape)
    return {"t": float(times[i]), "x": float(x_nodes[j]), "value": float(err[i, j])}


def timed(fn, repeats: int = 1):
    """Run ``fn`` ``repeats`` times; return (last_result, median_seconds)."""
    ts, res = [], None
    for _ in range(max(1, repeats)):
        t0 = time.perf_counter()
        res = fn()
        ts.append(time.perf_counter() - t0)
    return res, float(np.median(ts))
