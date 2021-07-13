import copy
import numpy as np

from pizza_cutter.des_pizza_cutter import (
    BMASK_SPLINE_INTERP, BMASK_GAIA_STAR
)

from ..stars import gen_gaia_mag_rad, mask_and_interp_stars
from ..constants import SIM_BMASK_GAIA_STAR


def test_gen_gaia_mag_rad():
    num = 5
    seed = 10
    rng = np.random.RandomState(seed=seed)
    mag_g, rad = gen_gaia_mag_rad(rng=rng, num=num)

    assert mag_g.shape == (5,)
    assert rad.shape == (5,)

    seed = 10
    rng = np.random.RandomState(seed=seed)
    mag_g1, rad1 = gen_gaia_mag_rad(rng=rng, num=num)

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
        "extra_noises": [
            rng.normal(size=(dim, dim)),
            rng.normal(size=(dim, dim)),
        ]
    }
    old_cdata = copy.deepcopy(cdata)

    rng = np.random.RandomState(seed=10)
    mask_and_interp_stars(
        rng=rng,
        cdata=cdata,
        stars=stars,
        interp_cfg={"iso_buff": 1, "fill_isolated_with_noise": False},
    )

    # check something changed
    for key in cdata:
        if key != "extra_noises":
            assert not np.array_equal(cdata[key], old_cdata[key])
        else:
            for i in range(len(cdata[key])):
                assert not np.array_equal(cdata[key][i], old_cdata[key][i])

    # check some sentinel values
    assert np.all(cdata["weight"][80:80, 70:80] == 0)
    assert np.all(cdata["mfrac"][80:80, 70:80] == 1)
    assert np.all((cdata["ormask"][80:80, 70:80] & SIM_BMASK_GAIA_STAR) != 0)
    assert np.all((cdata["bmask"][80:80, 70:80] & BMASK_GAIA_STAR) != 0)
    assert np.all((cdata["bmask"][80:80, 70:80] & BMASK_SPLINE_INTERP) != 0)

    # but not everything ofc
    assert not np.all(cdata["weight"] == 0)
    assert not np.all(cdata["mfrac"] == 1)
    assert not np.all((cdata["ormask"] & SIM_BMASK_GAIA_STAR) != 0)
    assert not np.all((cdata["bmask"] & BMASK_GAIA_STAR) != 0)
    assert not np.all((cdata["bmask"] & BMASK_SPLINE_INTERP) != 0)
