import copy
import numpy as np

import pytest

import galsim
from pizza_cutter.des_pizza_cutter import (
    BMASK_SPLINE_INTERP, BMASK_GAIA_STAR
)

from ..stars import (
    gen_gaia_mag_rad,
    mask_stars,
)
from ..mdet import make_mbobs_from_coadd_data


def test_gen_gaia_mag_rad():
    num = 5
    seed = 10
    rng = np.random.RandomState(seed=seed)
    mag_g, rad = gen_gaia_mag_rad(rng=rng, num=num, rad_dist="gaia+des")

    assert mag_g.shape == (5,)
    assert rad.shape == (5,)

    seed = 10
    rng = np.random.RandomState(seed=seed)
    mag_g1, rad1 = gen_gaia_mag_rad(rng=rng, num=num, rad_dist="gaia+des")

    assert np.array_equal(mag_g, mag_g1)
    assert np.array_equal(rad, rad1)


def test_mask_and_interp_stars():
    rng = np.random.RandomState(seed=11)
    stars = np.array(
        [(75, 85, 20)],
        dtype=[
            ("x", "f8"),
            ("y", "f8"),
            ("radius_pixels", "f8")
        ]
    )
    dim = 215

    cdata = {
        "image": rng.normal(size=(dim, dim)),
        "weight": np.ones((dim, dim)),
        "mfrac": np.zeros((dim, dim)),
        "ormask": np.zeros((dim, dim), dtype=np.int32),
        "bmask": np.zeros((dim, dim), dtype=np.int32),
        "noise": rng.normal(size=(dim, dim)),
        "psf": rng.normal(size=(dim, dim)),
    }

    mbobs = make_mbobs_from_coadd_data(
        wcs=galsim.PixelScale(0.263).jacobian(),
        cdata=cdata,
    )

    mbobs_old = copy.deepcopy(mbobs)

    rng = np.random.RandomState(seed=10)
    mask_stars(
        rng=rng,
        mbobs=mbobs,
        stars=stars,
        interp_cfg={"skip": False, "iso_buff": 1, "fill_isolated_with_noise": False},
        apodize_cfg={"skip": True, "ap_rad": 1},
        mask_expand_rad=0,
    )

    # check something changed
    for obslist, obslist_old in zip(mbobs, mbobs_old):
        for obs, obs_old in zip(obslist, obslist_old):
            for attr in ["image", "bmask", "noise", "weight", "mfrac"]:
                assert not np.array_equal(getattr(obs, attr), getattr(obs_old, attr))

    # check some sentinel values
    assert np.all(mbobs[0][0].weight[80:80, 70:80] == 0)
    assert np.all(mbobs[0][0].mfrac[80:80, 70:80] == 1)
    assert np.all((mbobs[0][0].bmask[80:80, 70:80] & BMASK_GAIA_STAR) != 0)
    assert np.all((mbobs[0][0].bmask[80:80, 70:80] & BMASK_SPLINE_INTERP) != 0)

    # but not everything ofc
    assert not np.all(mbobs[0][0].weight == 0)
    assert not np.all(mbobs[0][0].mfrac == 1)
    assert not np.all((mbobs[0][0].bmask & BMASK_GAIA_STAR) != 0)
    assert not np.all((mbobs[0][0].bmask & BMASK_SPLINE_INTERP) != 0)


def test_mask_and_apodize_stars():
    rng = np.random.RandomState(seed=11)
    stars = np.array(
        [(75, 85, 20)],
        dtype=[
            ("x", "f8"),
            ("y", "f8"),
            ("radius_pixels", "f8")
        ]
    )
    dim = 215

    cdata = {
        "image": rng.normal(size=(dim, dim)),
        "weight": np.ones((dim, dim)),
        "mfrac": np.zeros((dim, dim)),
        "ormask": np.zeros((dim, dim), dtype=np.int32),
        "bmask": np.zeros((dim, dim), dtype=np.int32),
        "noise": rng.normal(size=(dim, dim)),
        "psf": rng.normal(size=(dim, dim)),
    }

    mbobs = make_mbobs_from_coadd_data(
        wcs=galsim.PixelScale(0.263).jacobian(),
        cdata=cdata,
    )

    mbobs_old = copy.deepcopy(mbobs)

    rng = np.random.RandomState(seed=10)
    mask_stars(
        rng=rng,
        mbobs=mbobs,
        stars=stars,
        interp_cfg={"skip": True, "iso_buff": 1, "fill_isolated_with_noise": False},
        apodize_cfg={"skip": False, "ap_rad": 1},
        mask_expand_rad=0,
    )

    # check something changed
    for obslist, obslist_old in zip(mbobs, mbobs_old):
        for obs, obs_old in zip(obslist, obslist_old):
            for attr in ["image", "bmask", "noise", "weight", "mfrac"]:
                assert not np.array_equal(getattr(obs, attr), getattr(obs_old, attr))

    # check some sentinel values
    assert np.all(mbobs[0][0].weight[80:80, 70:80] == 0)
    assert np.all(mbobs[0][0].mfrac[80:80, 70:80] == 1)
    assert np.all((mbobs[0][0].bmask[80:80, 70:80] & BMASK_GAIA_STAR) != 0)
    assert np.all((mbobs[0][0].bmask[80:80, 70:80] & BMASK_SPLINE_INTERP) == 0)

    # but not everything ofc
    assert not np.all(mbobs[0][0].weight == 0)
    assert not np.all(mbobs[0][0].mfrac == 1)
    assert not np.all((mbobs[0][0].bmask & BMASK_GAIA_STAR) != 0)
    assert np.all((mbobs[0][0].bmask & BMASK_SPLINE_INTERP) == 0)


def test_mask_stars_raises():
    rng = np.random.RandomState(seed=11)
    stars = np.array(
        [(75, 85, 20)],
        dtype=[
            ("x", "f8"),
            ("y", "f8"),
            ("radius_pixels", "f8")
        ]
    )
    dim = 215

    cdata = {
        "image": rng.normal(size=(dim, dim)),
        "weight": np.ones((dim, dim)),
        "mfrac": np.zeros((dim, dim)),
        "ormask": np.zeros((dim, dim), dtype=np.int32),
        "bmask": np.zeros((dim, dim), dtype=np.int32),
        "noise": rng.normal(size=(dim, dim)),
        "psf": rng.normal(size=(dim, dim)),
    }

    mbobs = make_mbobs_from_coadd_data(
        wcs=galsim.PixelScale(0.263).jacobian(),
        cdata=cdata,
    )

    rng = np.random.RandomState(seed=10)
    with pytest.raises(RuntimeError):
        mask_stars(
            rng=rng,
            mbobs=mbobs,
            stars=stars,
            interp_cfg={
                "skip": False,
                "iso_buff": 1,
                "fill_isolated_with_noise": False
            },
            apodize_cfg={"skip": False, "ap_rad": 1},
            mask_expand_rad=0,
        )
