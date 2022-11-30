import galsim
import numpy as np

import pytest

from pizza_cutter_sims.psf import GalsimPSF, gen_psf


def test_psf_galsimpsf():
    obj = galsim.Gaussian(fwhm=0.8)
    psf = GalsimPSF(obj)
    assert psf.getPSF(None) is obj
    assert psf.getPSF(galsim.PositionD(10, 12)) is obj


def test_psf_gen_psf_gsobject():
    rng = np.random.RandomState(seed=42)
    psf_config = {
        "type": "galsim.Moffat",
        "beta": 2.5,
        "fwhm": 2.0,
        "fwhm_frac_std": 0,
        "shear": [0.1, 0.1],
        "shear_std": 0,
    }
    gs_config, psf = gen_psf(
        rng=rng, psf_config=psf_config, gal_config=None,
        se_image_shape=None, se_wcs=None
    )

    assert gs_config == {
        "type": "Moffat",
        "fwhm": 2.0,
        "shear": {"type": "G1G2", "g1": 0.1, "g2": 0.1},
        "beta": 2.5,
    }
    assert "Moffat" in repr(psf.gs_object), repr(psf.gs_object)
    assert "beta=2.5" in repr(psf.gs_object), repr(psf.gs_object)


def test_psf_gen_psf_gsobject_rng():
    rng = np.random.RandomState(seed=42)
    psf_config = {
        "type": "galsim.Moffat",
        "beta": 2.5,
        "fwhm": 2.0,
        "fwhm_frac_std": 0.1,
        "shear": [0.1, 0.1],
        "shear_std": 0.01,
    }
    gs_config, psf = gen_psf(
        rng=rng, psf_config=psf_config, gal_config=None,
        se_image_shape=None, se_wcs=None
    )

    rng = np.random.RandomState(seed=42)
    g1 = 0.1 + 0.01 * rng.normal()
    g2 = 0.1 + 0.01 * rng.normal()
    fwhm = 2.0 * (1 + 0.1 * rng.normal())

    assert gs_config == {
        "type": "Moffat",
        "fwhm": fwhm,
        "shear": {"type": "G1G2", "g1": g1, "g2": g2},
        "beta": 2.5,
    }
    assert "Moffat" in repr(psf.gs_object), repr(psf.gs_object)
    assert "beta=2.5" in repr(psf.gs_object), repr(psf.gs_object)

    rng = np.random.RandomState(seed=42)
    psf_config = {
        "type": "galsim.Moffat",
        "beta": 2.5,
        "fwhm": 2.0,
        "fwhm_frac_std": 0.1,
        "shear": [0.1, 0.1],
        "shear_std": 0.01,
    }
    gs_config1, psf1 = gen_psf(
        rng=rng, psf_config=psf_config, gal_config=None,
        se_image_shape=None, se_wcs=None
    )
    assert psf1.gs_object == psf.gs_object
    assert gs_config1 == gs_config


def test_psf_gen_psf_raises():
    rng = np.random.RandomState(seed=42)
    with pytest.raises(ValueError):
        gen_psf(
            rng=rng, psf_config={"type": "blah"}, gal_config=None,
            se_image_shape=None, se_wcs=None
        )


def test_psf_gen_psf_wldeblend_rng():
    rng = np.random.RandomState(seed=42)
    psf_config = {
        "type": "wldeblend",
        "fwhm": 2.0,
        "fwhm_frac_std": 0.1,
        "shear": [0.1, 0.1],
        "shear_std": 0.01,
    }
    gs_config, psf = gen_psf(
        rng=rng, psf_config=psf_config, gal_config={"type": "lsst-riz"},
        se_image_shape=None, se_wcs=None
    )

    rng = np.random.RandomState(seed=42)
    g1 = 0.1 + 0.01 * rng.normal()
    g2 = 0.1 + 0.01 * rng.normal()
    fwhm = 0.85 * (1 + 0.1 * rng.normal())

    assert gs_config == {
        "type": "Moffat",
        "fwhm": fwhm,
        "beta": 2.5,
        "shear": {"type": "G1G2", "g1": g1, "g2": g2},
    }
    assert "Moffat" in repr(psf.gs_object), repr(psf.gs_object)

    rng = np.random.RandomState(seed=42)
    psf_config = {
        "type": "wldeblend",
        "fwhm": 2.0,
        "fwhm_frac_std": 0.1,
        "shear": [0.1, 0.1],
        "shear_std": 0.01,
    }
    gs_config1, psf1 = gen_psf(
        rng=rng, psf_config=psf_config, gal_config={"type": "lsst-riz"},
        se_image_shape=None, se_wcs=None
    )
    assert psf1.gs_object == psf.gs_object
    assert gs_config1 == gs_config


def test_psf_gen_psf_wldeblend_ps_rng():
    rng = np.random.RandomState(seed=42)
    psf_config = {
        "type": "wldeblend-ps",
        "variation_factor": 8,
    }
    gs_config, psf = gen_psf(
        rng=rng, psf_config=psf_config, gal_config={"type": "lsst-riz"},
        se_image_shape=100, se_wcs=galsim.PixelScale(0.2),
    )
    gmod = psf.getPSF(galsim.PositionD(x=10, y=10))

    rng = np.random.RandomState(seed=42)
    psf_config = {
        "type": "wldeblend-ps",
        "variation_factor": 8,
    }
    gs_config1, psf1 = gen_psf(
        rng=rng, psf_config=psf_config, gal_config={"type": "lsst-riz"},
        se_image_shape=100, se_wcs=galsim.PixelScale(0.2),
    )
    gmod1 = psf1.getPSF(galsim.PositionD(x=10, y=10))

    assert gmod1 == gmod
    assert gs_config1 == gs_config

    assert gs_config["beta"] == 2.5
    assert gs_config["type"] == "Moffat"
    assert psf._variation_factor == 8
    assert psf._median_seeing == 0.85
