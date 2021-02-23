import copy

import numpy as np

from pizza_cutter_sims.tests.conftest import recursive_equal
from pizza_cutter_sims.sim import generate_sim


def test_generate_sim_seeding(sim_config):
    cfg = copy.deepcopy(sim_config)
    cfg["shear"]["g1"] = 0.0
    cfg["shear"]["g2"] = 0.0
    rng = np.random.RandomState(seed=42)
    gal_rng = np.random.RandomState(seed=42)
    sdata1 = generate_sim(
        rng=rng,
        gal_rng=gal_rng,
        coadd_config=cfg["coadd"],
        se_config=cfg["se"],
        psf_config=cfg["psf"],
        gal_config=cfg["gal"],
        layout_config=cfg["layout"],
        msk_config=cfg["msk"],
        shear_config=cfg["shear"],
    )

    cfg = copy.deepcopy(sim_config)
    cfg["shear"]["g1"] = 0.0
    cfg["shear"]["g2"] = 0.0
    rng = np.random.RandomState(seed=42)
    gal_rng = np.random.RandomState(seed=42)
    sdata2 = generate_sim(
        rng=rng,
        gal_rng=gal_rng,
        coadd_config=cfg["coadd"],
        se_config=cfg["se"],
        psf_config=cfg["psf"],
        gal_config=cfg["gal"],
        layout_config=cfg["layout"],
        msk_config=cfg["msk"],
        shear_config=cfg["shear"],
    )

    cfg = copy.deepcopy(sim_config)
    cfg["shear"]["g1"] = 0.0
    cfg["shear"]["g2"] = 0.0
    rng = np.random.RandomState(seed=4)
    gal_rng = np.random.RandomState(seed=42)
    sdata3 = generate_sim(
        rng=rng,
        gal_rng=gal_rng,
        coadd_config=cfg["coadd"],
        se_config=cfg["se"],
        psf_config=cfg["psf"],
        gal_config=cfg["gal"],
        layout_config=cfg["layout"],
        msk_config=cfg["msk"],
        shear_config=cfg["shear"],
    )

    assert recursive_equal(sdata1, sdata2)
    assert not recursive_equal(sdata1, sdata3)
