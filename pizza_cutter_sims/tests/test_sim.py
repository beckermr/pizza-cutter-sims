import copy

import numpy as np

from pizza_cutter_sims.sim import generate_sim


def _recursive_equal(sdata1, sdata2):
    eq = True
    if isinstance(sdata1, np.ndarray):
        eq = eq and np.array_equal(sdata1, sdata2)
    elif isinstance(sdata1, dict):
        for k in sdata1:
            eq = eq and _recursive_equal(sdata1[k], sdata2[k])
    elif isinstance(sdata1, list):
        for item1, item2 in zip(sdata1, sdata2):
            eq = eq and _recursive_equal(item1, item2)
    else:
        eq = eq and (sdata1 == sdata2)
    return eq


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

    assert _recursive_equal(sdata1, sdata2)
    assert not _recursive_equal(sdata1, sdata3)
