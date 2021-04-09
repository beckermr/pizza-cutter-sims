import numpy as np

from ..glass import make_glass_layout


def test_make_glass_layout():
    delta2 = (1/7.0)**2
    u, v = make_glass_layout(49, [0, 1], np.random.RandomState(seed=45))
    for _u, _v in zip(u, v):
        dr2 = (u - _u)**2 + (v - _v)**2
        msk = dr2 > 0
        assert np.all(dr2[msk] >= delta2)

    assert np.all(u >= 0)
    assert np.all(u <= 1)
    assert np.all(v >= 0)
    assert np.all(v <= 1)
    assert u.shape[0] > 0
    assert u.shape[0] == v.shape[0]
