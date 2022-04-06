import copy
import tempfile

import numpy as np

from pizza_cutter_sims.tests.conftest import recursive_equal
from pizza_cutter_sims.pizza_cutter import run_des_pizza_cutter_coadding_on_sim
from pizza_cutter_sims.sim import generate_sim


def _run_to_pizza_cutter(
    cfg, rng_seed, gal_rng_seed, coadd_rng_seed, star_rng_seed, return_sim=False
):
    cfg["shear"]["g1"] = 0.0
    cfg["shear"]["g2"] = 0.0

    rng = np.random.RandomState(seed=rng_seed)
    gal_rng = np.random.RandomState(seed=gal_rng_seed)
    star_rng = np.random.RandomState(seed=star_rng_seed)
    sdata = generate_sim(
        rng=rng,
        gal_rng=gal_rng,
        coadd_config=cfg["coadd"],
        se_config=cfg["se"],
        psf_config=cfg["psf"],
        gal_config=cfg["gal"],
        layout_config=cfg["layout"],
        msk_config=cfg["msk"],
        shear_config=cfg["shear"],
        star_rng=star_rng,
        star_config=cfg["star"],
        skip_coadding=False,
    )

    coadd_rng = np.random.RandomState(seed=coadd_rng_seed)
    with tempfile.TemporaryDirectory() as tmpdir:
        stars = sdata.pop("stars")
        psfs = sdata.pop("psfs")
        coadd_wcs = sdata.pop("coadd_wcs")
        sdata.pop("flux_zeropoints")
        cdata = run_des_pizza_cutter_coadding_on_sim(
            rng=coadd_rng,
            tmpdir=tmpdir,
            single_epoch_config=cfg["pizza_cutter"]["single_epoch_config"],
            n_extra_noise_images=0,
            **sdata,
        )
        sdata["stars"] = stars
        sdata["psfs"] = psfs
        sdata["coadd_wcs"] = coadd_wcs

    if return_sim:
        return cdata, sdata
    else:
        return cdata


def test_run_des_pizza_cutter_coadding_on_sim(sim_config):
    cdata, sdata = _run_to_pizza_cutter(
        copy.deepcopy(sim_config),
        42,
        43,
        44,
        45,
        return_sim=True,
    )

    # psf should be max at center of pixel
    msk = np.where(cdata["psf"] == np.max(cdata["psf"]))
    psf_cen = (cdata["psf"].shape[0] - 1)/2
    assert np.array_equal(msk, [[psf_cen], [psf_cen]])

    # coadd image should match se image since WCS is the same
    cdim = cdata["image"].shape[0]
    sdim = sdata["img"][0].shape[0]
    trim = (sdim - cdim) // 2
    shifts = []
    for shiftx in [-1, 0, 1]:
        for shifty in [-1, 0, 1]:
            diffim = (
                cdata["image"]
                - sdata["img"][0][trim-shifty:-trim-shifty, trim-shiftx:-trim-shiftx]
            )

            if shiftx == 0 and shifty == 0:
                max_shift = np.max(np.abs(diffim))
            else:
                shifts.append(np.max(np.abs(diffim)))

    assert all(s > max_shift for s in shifts)

    diffim = (
        cdata["image"]
        - sdata["img"][0][trim:-trim, trim:-trim]
    )
    # the tolerance is really high here due to the approximate WCS inverse used
    # by the coadding code. Making that wcs inverse more accurate by using more
    # sample points (delta -> 2) would allow this test to pass at 1e-6.
    if not np.allclose(
        cdata["image"],
        sdata["img"][0][trim:-trim, trim:-trim],
        atol=1e-4,
    ):
        import matplotlib.pyplot as plt
        plt.figure()
        plt.imshow(diffim[:20, :20])
        assert False, np.max(np.abs(diffim))


def test_run_des_pizza_cutter_coadding_on_sim_seeding(sim_config):
    cdata1 = _run_to_pizza_cutter(
        copy.deepcopy(sim_config),
        42,
        43,
        44,
        45,
    )

    cdata2 = _run_to_pizza_cutter(
        copy.deepcopy(sim_config),
        42,
        43,
        44,
        45,
    )
    assert recursive_equal(cdata1, cdata2)

    for seed_set in [(52, 43, 44, 45), (42, 53, 44, 47), (42, 43, 54, 67)]:
        cdata3 = _run_to_pizza_cutter(
            copy.deepcopy(sim_config),
            *seed_set,
        )
        assert not recursive_equal(cdata1, cdata3)
