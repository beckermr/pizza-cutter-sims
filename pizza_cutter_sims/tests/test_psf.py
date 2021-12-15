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
    gs_config, psf = gen_psf(rng=rng, psf_config=psf_config)

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
    gs_config, psf = gen_psf(rng=rng, psf_config=psf_config)

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


def test_psf_gen_psf_raises():
    rng = np.random.RandomState(seed=42)
    with pytest.raises(ValueError):
        gen_psf(rng=rng, psf_config={"type": "blah"})
