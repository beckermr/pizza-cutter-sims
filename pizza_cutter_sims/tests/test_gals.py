import numpy as np

import pytest

from pizza_cutter_sims.gals import gen_gals


def test_gals_gen_gals_grid():
    rng = np.random.RandomState(seed=42)
    layout_config = {
        "type": "grid",
        "ngal_per_side": 7,
        "dither_scale": 0.263,
    }
    pos_bounds = (-10, 10)
    gal_config = {
        "type": "exp-bright",
        "noise": 10,
    }
    gals, upos, vpos, noise, noise_scale = gen_gals(
        rng=rng,
        layout_config=layout_config,
        pos_bounds=pos_bounds,
        gal_config=gal_config,
    )
    assert len(gals) == upos.shape[0]
    assert len(gals) == vpos.shape[0]
    assert len(gals) == 49
    assert noise == 10
    assert all(["Sersic" in repr(g) for g in gals])
    assert np.all(
        (upos >= -10) &
        (upos <= 10) &
        (vpos >= -10) &
        (upos <= 10)
    )
    assert noise_scale is None


def test_gals_gen_gals_random():
    rng = np.random.RandomState(seed=42)
    layout_config = {
        "type": "random",
        "ngal_per_arcmin2": 60,
        "dither_scale": 0.263,
    }
    pos_bounds = (-10, 10)
    gal_config = {
        "type": "exp-bright",
        "noise": 10,
    }
    gals, upos, vpos, noise, noise_scale = gen_gals(
        rng=rng,
        layout_config=layout_config,
        pos_bounds=pos_bounds,
        gal_config=gal_config,
    )
    assert len(gals) == upos.shape[0]
    assert len(gals) == vpos.shape[0]
    assert noise == 10
    assert all(["Sersic" in repr(g) for g in gals])
    assert np.all(
        (upos >= -10) &
        (upos <= 10) &
        (vpos >= -10) &
        (upos <= 10)
    )
    assert noise_scale is None


def test_gals_gen_gals_raises():
    rng = np.random.RandomState(seed=42)
    with pytest.raises(ValueError):
        gen_gals(
            rng=rng,
            layout_config=None,
            gal_config={"type": "blah"},
            pos_bounds=(-10, 10),
        )


def test_gals_gen_gals_wldeblend_des():
    rng = np.random.RandomState(seed=42)
    layout_config = {
        "type": "random",
        "ngal_per_arcmin2": 60,
        "dither_scale": 0.263,
    }
    pos_bounds = (-10, 10)
    gal_config = {
        "type": "des-riz",
        "noise": 10,
    }
    gals, upos, vpos, noise, noise_scale = gen_gals(
        rng=rng,
        layout_config=layout_config,
        pos_bounds=pos_bounds,
        gal_config=gal_config,
    )
    assert len(gals) == upos.shape[0]
    assert len(gals) == vpos.shape[0]
    assert all(["galsim.Sum" in repr(g) for g in gals])
    assert np.all(
        (upos >= -10) &
        (upos <= 10) &
        (vpos >= -10) &
        (upos <= 10)
    )
    assert noise_scale == 0.263


def test_gals_gen_gals_wldeblend_lsst():
    rng = np.random.RandomState(seed=42)
    layout_config = {
        "type": "random",
        "ngal_per_arcmin2": 60,
        "dither_scale": 0.263,
    }
    pos_bounds = (-10, 10)
    gal_config = {
        "type": "lsst-riz",
        "noise": 10,
    }
    gals, upos, vpos, noise, noise_scale = gen_gals(
        rng=rng,
        layout_config=layout_config,
        pos_bounds=pos_bounds,
        gal_config=gal_config,
    )
    assert len(gals) == upos.shape[0]
    assert len(gals) == vpos.shape[0]
    assert all(["galsim.Sum" in repr(g) for g in gals])
    assert np.all(
        (upos >= -10) &
        (upos <= 10) &
        (vpos >= -10) &
        (upos <= 10)
    )
    assert noise_scale == 0.2
