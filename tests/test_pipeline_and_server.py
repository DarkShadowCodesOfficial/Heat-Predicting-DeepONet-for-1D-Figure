import time

import numpy as np
import pytest
from fastapi.testclient import TestClient

from heat_operator.config import SENSORS, get_preset
from heat_operator.persistence import load_bundle


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    from demo.server import create_app
    ckpt = tmp_path_factory.mktemp("ckpt")
    app = create_app(preset="tiny", ckpt_root=str(ckpt))
    with TestClient(app) as c:
        for _ in range(240):
            s = c.get("/api/status").json()
            if s["state"] in ("ready", "error"):
                break
            time.sleep(0.5)
        assert s["state"] == "ready", s
        c.ckpt = ckpt
        yield c


def _ic():
    ic = np.full(SENSORS, 22.0)
    ic[0] = 100.0
    ic[8:12] += 25
    return ic.tolist()


def test_checkpoint_roundtrip(client):
    b = load_bundle(client.ckpt / "tiny", get_preset("tiny"))
    assert b is not None and len(b.linear_models) == get_preset("tiny").ens_n
    assert load_bundle(client.ckpt / "tiny", get_preset("quick")) is None   # config mismatch


def test_config_endpoint(client):
    c = client.get("/api/config").json()
    assert c["sensors"] == SENSORS and len(c["x_nodes"]) == 101
    assert c["training"]["al_history"]


@pytest.mark.parametrize("mode", ["linear", "nonlinear"])
def test_simulate(client, mode):
    body = {"ic": _ic(), "alpha": 1.2e-3, "mode": mode, "t": 600}
    r = client.post("/api/simulate", json=body)
    assert r.status_code == 200, r.text
    d = r.json()
    nt, nx = len(d["times"]), len(d["x"])
    assert np.array(d["fem_grid"]).shape == (nt, nx)
    assert np.array(d["op_grid"]).shape == (nt, nx)
    assert np.array(d["err_grid"]).shape == (nt, nx)
    assert d["snapshot"]["t"] == 600
    assert len(d["intervals"]["matrix"]) == 5
    assert d["timing"]["fem_full_s"] > 0 and d["timing"]["op_full_s"] > 0
    assert d["has_uncertainty"] == (mode == "linear")


def test_snapshot_and_analytical(client):
    ic = np.full(SENSORS, 20.0); ic[0] = 100.0
    body = {"ic": ic.tolist(), "alpha": 1e-3, "mode": "linear", "t": 900, "reference": "analytical"}
    d = client.post("/api/simulate", json=body).json()
    assert d["analytical_available"] and d["reference"] == "analytical"
    assert d["fem_vs_analytical"]["mae"] < 0.5
    s = client.post("/api/snapshot", json=body).json()
    assert s["snapshot"]["t"] == 900 and "fem_s" in s["snapshot"]["timing"]


def test_validation(client):
    r = client.post("/api/simulate", json={"ic": [1, 2, 3], "alpha": 1e-3, "mode": "linear", "t": 0})
    assert r.status_code == 422
    r = client.get("/api/random_ic?seed=3").json()
    assert len(r["ic"]) == SENSORS and 5e-4 <= r["alpha"] <= 2.5e-3
