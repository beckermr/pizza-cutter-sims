import numpy as np

from ..masking import generate_bad_columns, generate_cosmic_rays


def test_generate_cosmic_rays_seed():
    rng = np.random.RandomState(seed=10)
    msk1 = generate_cosmic_rays((64, 64), rng=rng)

    rng = np.random.RandomState(seed=10)
    msk2 = generate_cosmic_rays((64, 64), rng=rng)

    assert np.array_equal(msk1, msk2)


def test_generate_bad_columns_seed():
    rng = np.random.RandomState(seed=10)
    msk1 = generate_bad_columns((64, 64), rng=rng)

    rng = np.random.RandomState(seed=10)
    msk2 = generate_bad_columns((64, 64), rng=rng)

    assert np.array_equal(msk1, msk2)
