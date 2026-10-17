import numpy as np
import torch

from heat_operator.config import SENSORS
from heat_operator.data import convert, extract, rand_ic, rand_mat
from heat_operator.model import DeepONet
from heat_operator.physics import physics_loss
from heat_operator.training import train1


def test_forward_shape():
    m = DeepONet()
    out = m(torch.rand(7, SENSORS), torch.full((7,), 1e-3), torch.rand(7), torch.rand(7) * 3600)
    assert out.shape == (7,)


def test_physics_losses_backprop():
    rng = np.random.default_rng(0)
    ic, a = rand_ic(rng), rand_mat(rng)
    m = DeepONet()
    for beta in (None, -0.003):
        loss = physics_loss(m, ic, a, beta=beta, n_phys=16)
        assert torch.isfinite(loss)
        loss.backward()


def test_training_reduces_data_loss():
    rng = np.random.default_rng(0)
    pool = [(rand_ic(rng), rand_mat(rng)) for _ in range(3)]
    egs = [e for ic, a in pool for e in extract(ic, a, rng, 40)]
    torch.manual_seed(0)
    m = DeepONet()
    losses = []
    train1(m, egs, pool, 60, callback=lambda e, n, dl, pl: losses.append(dl))
    assert np.mean(losses[-5:]) < np.mean(losses[:5])


def test_shared_inference_matches_forward():
    rng = np.random.default_rng(0)
    ic, a = rand_ic(rng), rand_mat(rng)
    m = DeepONet()
    x, t = torch.rand(50), torch.rand(50) * 3600
    ref = m(torch.tensor(np.tile(ic, (50, 1)), dtype=torch.float32), torch.full((50,), a), x, t)
    out = m.forward_shared(torch.tensor(ic, dtype=torch.float32), a, x, t)
    torch.testing.assert_close(out, ref.detach(), atol=1e-4, rtol=1e-4)
