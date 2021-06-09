import numpy as np

from pizza_cutter_sims.masking import (
    generate_bad_columns,
    generate_cosmic_rays,
    generate_streaks,
)


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


def test_generate_streaks_seed():
    rng = np.random.RandomState(seed=10)
    msk1 = generate_streaks((64, 64), rng=rng)

    rng = np.random.RandomState(seed=10)
    msk2 = generate_streaks((64, 64), rng=rng)

    assert np.array_equal(msk1, msk2)
