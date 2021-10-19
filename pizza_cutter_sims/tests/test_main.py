import copy

from pizza_cutter_sims.main import (
    run_end2end_with_shear,
    run_end2end_pair_with_shear,
)
from pizza_cutter_sims.tests.conftest import recursive_equal


def test_run_end2end_with_shear_seeding(sim_config):
    seed_kwargs = dict(
        rng_seed=42,
        gal_rng_seed=43,
        coadd_rng_seed=44,
        mdet_rng_seed=45,
        star_rng_seed=46,
    )
    res1 = run_end2end_with_shear(
        cfg=copy.deepcopy(sim_config),
        g1=0.02,
        g2=0.0,
        **seed_kwargs,
    )

    res2 = run_end2end_with_shear(
        cfg=copy.deepcopy(sim_config),
        g1=0.02,
        g2=0.0,
        **seed_kwargs,
    )

    assert recursive_equal(res1, res2)

    for k in seed_kwargs:
        _kws = copy.deepcopy(seed_kwargs)
        _kws[k] = 100
        res3 = run_end2end_with_shear(
            cfg=copy.deepcopy(sim_config),
            g1=0.02,
            g2=0.0,
            **_kws,
        )
        assert not recursive_equal(res1, res3), k


def test_run_end2end_pair_with_shear_seeding(sim_config):
    seed_kwargs = dict(
        rng_seed=42,
        gal_rng_seed=43,
        coadd_rng_seed=44,
        mdet_rng_seed=45,
        star_rng_seed=46,
    )
    res1 = run_end2end_pair_with_shear(
        cfg=copy.deepcopy(sim_config),
        g1=0.02,
        g2=0.0,
        swap12=False,
        **seed_kwargs,
    )

    res2 = run_end2end_pair_with_shear(
        cfg=copy.deepcopy(sim_config),
        g1=0.02,
        g2=0.0,
        swap12=False,
        **seed_kwargs,
    )

    assert recursive_equal(res1, res2)

    for k in seed_kwargs:
        _kws = copy.deepcopy(seed_kwargs)
        _kws[k] = 100
        res3 = run_end2end_pair_with_shear(
            cfg=copy.deepcopy(sim_config),
            g1=0.02,
            g2=0.0,
            swap12=False,
            **_kws,
        )
        assert not recursive_equal(res1, res3), k


def test_run_end2end_with_shear_skip_coadding(sim_config):
    seed_kwargs = dict(
        rng_seed=42,
        gal_rng_seed=43,
        coadd_rng_seed=44,
        mdet_rng_seed=45,
        star_rng_seed=46,
    )
    cfg = copy.deepcopy(sim_config)
    cfg["pizza_cutter"]["skip"] = True
    run_end2end_with_shear(
        cfg=cfg,
        g1=0.02,
        g2=0.0,
        **seed_kwargs,
    )
