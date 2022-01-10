import numpy as np

import galsim

from pizza_cutter_sims.gals_wldeblend import (
    init_wldeblend,
    get_gal_wldeblend,
    get_psf_config_wldeblend
)


def test_init_wldeblend():
    data1 = init_wldeblend(
        survey_bands="lsst-riz",
    )

    data2 = init_wldeblend(
        survey_bands="lsst-riz",
    )
    assert data1 is data2

    data3 = init_wldeblend(
        survey_bands="des-ri",
    )
    assert data1 is not data3

    assert data1.psf_fwhm < data3.psf_fwhm
    assert len(data1.builders) == 3
    assert len(data1.surveys) == 3
    assert data1.survey_name == "lsst"
    assert data1.ngal_per_arcmin2 > 0
    assert len(data1.cat) > 0
    assert data1.pixel_scale == 0.2

    assert len(data3.builders) == 2
    assert len(data3.surveys) == 2
    assert data3.survey_name == "des"
    assert data3.ngal_per_arcmin2 > 0
    assert len(data3.cat) > 0
    assert data3.pixel_scale == 0.263


def test_get_gal_wldeblend():
    data = init_wldeblend(
        survey_bands="lsst-riz",
    )

    rng = np.random.RandomState(seed=10)
    gal1 = get_gal_wldeblend(
        rng=rng,
        data=data,
    )

    rng = np.random.RandomState(seed=10)
    gal2 = get_gal_wldeblend(
        rng=rng,
        data=data,
    )

    gal3 = get_gal_wldeblend(
        rng=rng,
        data=data,
    )

    assert repr(gal1) == repr(gal2)
    assert repr(gal1) != repr(gal3)

    assert isinstance(gal1, galsim.GSObject)


def test_get_psf_config_wldeblend():
    data = init_wldeblend(
        survey_bands="lsst-riz",
    )
    assert get_psf_config_wldeblend(data=data) == {"type": "Kolmogorov", "fwhm": 0.85}

    data = init_wldeblend(
        survey_bands="des-ri",
    )
    assert get_psf_config_wldeblend(data=data) == {"type": "Kolmogorov", "fwhm": 1.1}
