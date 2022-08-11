import copy

import numpy as np

import pytest

from pizza_cutter_sims.tests.conftest import recursive_equal
from pizza_cutter_sims.sim import generate_sim


def _run_sim(cfg, rng_seed, gal_rng_seed, star_rng_seed):
    cfg["shear"]["g1"] = 0.0
    cfg["shear"]["g2"] = 0.0

    rng = np.random.RandomState(seed=rng_seed)
    gal_rng = np.random.RandomState(seed=gal_rng_seed)
    star_rng = np.random.RandomState(seed=star_rng_seed)
    return generate_sim(
        rng=rng,
        gal_rng=gal_rng,
        coadd_config=cfg["coadd"],
        se_config=cfg["se"],
        psf_config=cfg["psf"],
        gal_config=cfg["gal"],
        layout_config=cfg["layout"],
        msk_config=cfg["msk"],
        shear_config=cfg["shear"],
        star_config=cfg["star"],
        star_rng=star_rng,
        skip_coadding=cfg["pizza_cutter"]["skip"],
    )


def test_generate_sim_seeding(sim_config):
    sdata1 = _run_sim(copy.deepcopy(sim_config), 42, 43, 45)
    sdata2 = _run_sim(copy.deepcopy(sim_config), 42, 43, 45)

    assert recursive_equal(sdata1, sdata2)

    for seed_set in [(52, 43, 34), (42, 53, 67)]:
        sdata3 = _run_sim(copy.deepcopy(sim_config), *seed_set)
        assert not recursive_equal(sdata1, sdata3)


@pytest.mark.parametrize("key,val", [
    ("position_angle_range", [0, 360]),
    ("dither_scale", 3),
    ("scale_frac_std", 0.1),
    ("shear_std", 0.3),
])
def test_generate_sim_wcs_changes(sim_config, key, val):
    sdata1 = _run_sim(copy.deepcopy(sim_config), 42, 43, 45)
    sconfig = copy.deepcopy(sim_config)
    sconfig["se"]["wcs_config"][key] = val
    sdata2 = _run_sim(sconfig, 42, 43, 45)

    assert not recursive_equal(sdata1, sdata2)


def test_generate_sim_wldeblend(sim_config):
    cfg = copy.deepcopy(sim_config)
    cfg["gal"] = {
        "type": "lsst-riz",
        "multiband": False,
    }
    sdata1 = _run_sim(cfg, 42, 43, 45)
    sdata2 = _run_sim(cfg, 42, 43, 45)
    assert recursive_equal(sdata1, sdata2)


def test_generate_sim_wldeblend_multiband(sim_config):
    cfg = copy.deepcopy(sim_config)
    cfg["gal"] = {
        "type": "lsst-riz",
        "multiband": True,
    }
    cfg["se"]["n_images"] = 3
    cfg["pizza_cutter"]["skip"] = True
    cfg["psf"]["color_range"] = [0, 3]
    cfg["psf"]["dilation_range"] = [0.8, 1.1]
    sdata1 = _run_sim(cfg, 42, 43, 45)
    sdata2 = _run_sim(cfg, 42, 43, 45)
    assert recursive_equal(sdata1, sdata2)

    cfg["psf"]["dilation_range"] = [1, 1]
    sdata2 = _run_sim(cfg, 42, 43, 45)

    assert not recursive_equal(sdata1, sdata2)
