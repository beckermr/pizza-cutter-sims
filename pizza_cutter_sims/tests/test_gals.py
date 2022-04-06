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
        "multiband": False,
        "color_range": [1, 1],
    }
    gals, upos, vpos, noise, noise_scale, colors, _ = gen_gals(
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
    assert np.all(c == 1 for c in colors)


def test_gals_gen_gals_grid_multiband():
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
        "multiband": 5,
        "color_range": [0, 0],
    }
    gals, upos, vpos, noise, noise_scale, colors, flux_zeropoints = gen_gals(
        rng=rng,
        layout_config=layout_config,
        pos_bounds=pos_bounds,
        gal_config=gal_config,
    )
    assert len(gals) == upos.shape[0]
    assert len(gals) == vpos.shape[0]
    assert len(gals) == 49
    assert noise == 10
    assert all([all(["Sersic" in repr(g) for g in gal]) for gal in gals])
    assert np.all(
        (upos >= -10) &
        (upos <= 10) &
        (vpos >= -10) &
        (upos <= 10)
    )
    assert noise_scale is None
    assert np.all(c == 0 for c in colors)
    flux = gals[0][0].flux
    assert all([all([g.flux == flux for g in gal]) for gal in gals])
    for gal in gals:
        color = flux_zeropoints[0] - np.log10(gal[0].flux)/0.4
        color -= (flux_zeropoints[1] - np.log10(gal[1].flux)/0.4)
        assert color == 0


def test_gals_gen_gals_grid_multiband_color():
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
        "multiband": 5,
        "color_range": [1, 1],
    }
    gals, upos, vpos, noise, noise_scale, colors, flux_zeropoints = gen_gals(
        rng=rng,
        layout_config=layout_config,
        pos_bounds=pos_bounds,
        gal_config=gal_config,
    )
    assert len(gals) == upos.shape[0]
    assert len(gals) == vpos.shape[0]
    assert len(gals) == 49
    assert noise == 10
    assert all([all(["Sersic" in repr(g) for g in gal]) for gal in gals])
    assert np.all(
        (upos >= -10) &
        (upos <= 10) &
        (vpos >= -10) &
        (upos <= 10)
    )
    assert noise_scale is None
    assert np.all(c == 0 for c in colors)
    flux0 = gals[0][0].flux
    flux1 = gals[0][1].flux
    assert flux0 != flux1
    for gal in gals:
        assert gal[0].flux == flux0
        assert gal[1].flux == flux1
        assert all(g.flux == flux0 for g in gal[2:])
        color = flux_zeropoints[0] - np.log10(gal[0].flux)/0.4
        color -= (flux_zeropoints[1] - np.log10(gal[1].flux)/0.4)
        assert color == 1


def test_gals_gen_gals_grid_multiband_color_range():
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
        "multiband": 5,
        "color_range": [1, 4],
    }
    gals, upos, vpos, noise, noise_scale, colors, flux_zeropoints = gen_gals(
        rng=rng,
        layout_config=layout_config,
        pos_bounds=pos_bounds,
        gal_config=gal_config,
    )
    assert len(gals) == upos.shape[0]
    assert len(gals) == vpos.shape[0]
    assert len(gals) == 49
    assert noise == 10
    assert all([all(["Sersic" in repr(g) for g in gal]) for gal in gals])
    assert np.all(
        (upos >= -10) &
        (upos <= 10) &
        (vpos >= -10) &
        (upos <= 10)
    )
    assert noise_scale is None
    assert np.all(c == 0 for c in colors)
    for gal, true_color in zip(gals, colors):
        flux0 = gal[0].flux
        flux1 = gal[1].flux
        assert flux0 != flux1
        assert all(g.flux == flux0 for g in gal[2:])
        color = flux_zeropoints[0] - np.log10(gal[0].flux)/0.4
        color -= (flux_zeropoints[1] - np.log10(gal[1].flux)/0.4)
        assert np.allclose(color, true_color)


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
        "multiband": False,
        "color_range": [1, 2],
    }
    gals, upos, vpos, noise, noise_scale, colors, _ = gen_gals(
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
    assert np.mean(colors) > 1
    assert np.mean(colors) < 2
    assert all(c > 1 and c < 2 for c in colors)


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
    gals, upos, vpos, noise, noise_scale, colors, _ = gen_gals(
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
    assert all(c > -5 for c in colors)
    assert all(c < 5 for c in colors)
    assert np.median(colors) > 0


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
    gals, upos, vpos, noise, noise_scale, colors, _ = gen_gals(
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
    assert all(c > -5 for c in colors)
    assert all(c < 5 for c in colors)
    assert np.median(colors) > 0
