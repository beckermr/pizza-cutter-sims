import numpy as np
import galsim

from pizza_cutter_sims.ps_psf import PowerSpectrumPSF
from pizza_cutter_sims.constants import PIXEL_SCALE
from pizza_cutter_sims.gsutils import build_gsobject


def _get_fwhm_g1g2(psf_im):
    mom = galsim.hsm.FindAdaptiveMom(psf_im)
    return (
        mom.moments_sigma * PIXEL_SCALE * 2.355,
        mom.observed_shape.g1,
        mom.observed_shape.g2)


def test_ps_psf_smoke():
    rng = np.random.RandomState(seed=10)
    ps = PowerSpectrumPSF(
        rng=rng,
        im_width=120,
        buff=20,
        scale=PIXEL_SCALE,
        trunc=10)
    psf = ps.getPSF(galsim.PositionD(x=10, y=10))
    assert isinstance(psf, galsim.GSObject)
    psf_im = psf.drawImage(scale=PIXEL_SCALE)
    assert psf_im.calculateFWHM() > 0.5


def test_ps_psf_seeding():
    ps1 = PowerSpectrumPSF(
        rng=np.random.RandomState(seed=10),
        im_width=120,
        buff=20,
        scale=PIXEL_SCALE,
        trunc=10)
    ps2 = PowerSpectrumPSF(
        rng=np.random.RandomState(seed=10),
        im_width=120,
        buff=20,
        scale=PIXEL_SCALE,
        trunc=10)

    psf1 = ps1.getPSF(galsim.PositionD(x=10, y=10))
    psf_im1 = psf1.drawImage(scale=PIXEL_SCALE)

    psf2 = ps2.getPSF(galsim.PositionD(x=10, y=10))
    psf_im2 = psf2.drawImage(scale=PIXEL_SCALE)

    assert np.array_equal(psf_im1.array, psf_im2.array)


def test_ps_psf_variation():
    rng = np.random.RandomState(seed=10)
    ps = PowerSpectrumPSF(
        rng=rng,
        im_width=120,
        buff=20,
        scale=PIXEL_SCALE,
        trunc=1)

    psf1 = ps.getPSF(galsim.PositionD(x=0, y=0))
    psf_im1 = psf1.drawImage(scale=PIXEL_SCALE)
    fwhm1, g11, g21 = _get_fwhm_g1g2(psf_im1)

    psf2 = ps.getPSF(galsim.PositionD(x=119, y=119))
    psf_im2 = psf2.drawImage(scale=PIXEL_SCALE)
    fwhm2, g12, g22 = _get_fwhm_g1g2(psf_im2)

    assert not np.array_equal(psf_im1.array, psf_im2.array)
    assert np.abs(fwhm1/fwhm2 - 1) > 0.002
    assert np.abs(g11/g12 - 1) > 0.01
    assert np.abs(g21/g22 - 1) > 0.01


def test_ps_psf_truncation():
    ps1 = PowerSpectrumPSF(
        rng=np.random.RandomState(seed=15),
        im_width=120,
        buff=120,
        scale=PIXEL_SCALE,
        trunc=1)
    ps2 = PowerSpectrumPSF(
        rng=np.random.RandomState(seed=15),
        im_width=120,
        buff=120,
        scale=PIXEL_SCALE,
        trunc=100)

    g1s1 = []
    g2s1 = []
    g1s2 = []
    g2s2 = []
    for x in np.linspace(0, 119, 10):
        for y in np.linspace(0, 119, 10):
            psf1 = ps1.getPSF(galsim.PositionD(x=x, y=y))
            psf_im1 = psf1.drawImage(scale=PIXEL_SCALE)
            _, g1, g2 = _get_fwhm_g1g2(psf_im1)
            g1s1.append(g1)
            g2s1.append(g2)

            psf2 = ps2.getPSF(galsim.PositionD(x=x, y=y))
            psf_im2 = psf2.drawImage(scale=PIXEL_SCALE)
            _, g1, g2 = _get_fwhm_g1g2(psf_im2)
            g1s2.append(g1)
            g2s2.append(g2)

    assert np.std(g1s1) > np.std(g1s2)
    assert np.std(g2s1) > np.std(g2s2)


def test_ps_psf_config():
    rng = np.random.RandomState(seed=10)
    ps = PowerSpectrumPSF(
        rng=rng,
        im_width=120,
        buff=20,
        scale=PIXEL_SCALE,
        trunc=10)
    psf = ps.getPSF(galsim.PositionD(x=10, y=10))
    gs_config = ps.getPSFConfig(galsim.PositionD(x=10, y=10))
    assert isinstance(psf, galsim.GSObject)
    assert isinstance(gs_config, dict)

    config_psf = build_gsobject(config=gs_config, kind='psf')
    # galsim Moffat profiles support this
    assert config_psf == psf
