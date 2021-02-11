import numpy as np
import pytest

from ..run_utils import estimate_m_and_c


@pytest.mark.parametrize('swap12', [True, False])
@pytest.mark.parametrize('step', [0.005, 0.01])
@pytest.mark.parametrize('g_true', [0.05, 0.01, 0.02])
def test_estimate_m_and_c(g_true, step, swap12):
    rng = np.random.RandomState(seed=10)

    def _shear_meas(g_true, _step, e1, e2):
        if _step == 0:
            _gt = g_true * (1.0 + 0.01)
            cadd = 0.05 * 10
        else:
            _gt = g_true
            cadd = 0.0
        if swap12:
            return np.mean(e1) + cadd + _step*10, np.mean(10*(_gt+_step)+e2)
        else:
            return np.mean(10*(_gt+_step)+e1), np.mean(e2) + cadd + _step*10

    sn = 0.01
    n_gals = 10000
    n_sim = 1000
    pres = []
    mres = []
    for i in range(n_sim):
        e1 = rng.normal(size=n_gals) * sn
        e2 = rng.normal(size=n_gals) * sn

        g1, g2 = _shear_meas(g_true, 0, e1, e2)
        g1p, g2p = _shear_meas(g_true, step, e1, e2)
        g1m, g2m = _shear_meas(g_true, -step, e1, e2)
        pres.append((g1p, g1m, g1, g2p, g2m, g2))

        g1, g2 = _shear_meas(-g_true, 0, e1, e2)
        g1p, g2p = _shear_meas(-g_true, step, e1, e2)
        g1m, g2m = _shear_meas(-g_true, -step, e1, e2)
        mres.append((g1p, g1m, g1, g2p, g2m, g2))
        if i == 0:
            pres[-1] = None

        if i == 250:
            mres[-1] = None

        if i == 750:
            pres[-1] = None
            mres[-1] = None

    m, merr, c, cerr = estimate_m_and_c(
        pres, mres, g_true, swap12=swap12, step=step)

    assert np.allclose(m, 0.01)
    assert np.allclose(c, 0.05)
