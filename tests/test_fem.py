import numpy as np
from scipy.linalg import lu_factor, lu_solve

from heat_operator import config as C
from heat_operator.analytical import analytical_field, uniform_ic_value
from heat_operator.data import rand_ic, rand_mat
from heat_operator.fem import (K_LOCAL, M, assemble_K_nonlinear, alpha_of_T,
                               run_fem, run_fem_nonlinear)


def notebook_assemble_K_nonlinear(T_field, alpha0, beta, T_ref=20.0):
    """Verbatim from the notebook (python loops)."""
    K_nl = np.zeros((C.NUM_NODES, C.NUM_NODES))
    for e in range(C.NUM_NODES - 1):
        nodes = [e, e + 1]
        T_avg = 0.5 * (T_field[nodes[0]] + T_field[nodes[1]])
        alpha_e = alpha_of_T(T_avg, alpha0, beta, T_ref)
        for a in range(2):
            for b in range(2):
                K_nl[nodes[a], nodes[b]] += alpha_e * K_local_nb[a, b]
    return K_nl


K_local_nb = K_LOCAL


def test_vectorised_assembly_matches_notebook():
    rng = np.random.default_rng(0)
    T = rand_ic(rng)
    T = np.interp(C.X_NODES, C.SENSOR_X, T)
    np.testing.assert_allclose(assemble_K_nonlinear(T, 1e-3, -0.003),
                               notebook_assemble_K_nonlinear(T, 1e-3, -0.003), atol=1e-12)


def test_early_stop_matches_full_run():
    rng = np.random.default_rng(1)
    ic, a = rand_ic(rng), rand_mat(rng)
    full = run_fem(ic, a, [0.0, 120.0, 3600.0])
    part = run_fem(ic, a, [0.0, 120.0])
    np.testing.assert_allclose(full[120.0], part[120.0], atol=1e-12)


def test_nonlinear_reduces_to_linear_when_beta_zero():
    rng = np.random.default_rng(2)
    ic, a = rand_ic(rng), rand_mat(rng)
    lin = run_fem(ic, a, [300.0])[300.0]
    nl, _ = run_fem_nonlinear(ic, a, 0.0, [300.0])
    np.testing.assert_allclose(lin, nl[300.0], atol=1e-6)


def test_fem_matches_analytical_for_uniform_ic():
    ic = np.full(C.SENSORS, 20.0)
    ic[0] = C.HEATER
    alpha = 1.5e-3
    times = [300.0, 1200.0, 3600.0]
    fem = run_fem(ic, alpha, times)
    ana = analytical_field(C.X_NODES, times, alpha, 20.0)
    mae = np.mean([np.mean(np.abs(fem[t] - ana[i])) for i, t in enumerate(times)])
    assert mae < 0.5, mae          # coarse sensor interpolation of the heater step


def test_boundary_and_bounds():
    rng = np.random.default_rng(3)
    ic, a = rand_ic(rng), rand_mat(rng)
    out = run_fem(ic, a, [10.0, 3600.0])
    for T in out.values():
        assert abs(T[0] - C.HEATER) < 1e-9
    assert out[3600.0].min() > 15.0       # heating toward 100


def test_uniform_detection():
    ic = np.full(C.SENSORS, 25.0); ic[0] = 100.0
    assert uniform_ic_value(ic) == 25.0
    ic[5] = 30.0
    assert uniform_ic_value(ic) is None
