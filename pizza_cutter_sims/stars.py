import logging
import numpy as np

from pizza_cutter.des_pizza_cutter import (
    BMASK_SPLINE_INTERP, BMASK_GAIA_STAR
)
from metadetect.masking import apply_foreground_masking_corrections

GAIA_STAR_DIST_POLY = [
    3.7145287223147343e-09,
    9.522190086985166e-07,
    -5.692480432853031e-05,
    0.001281440711276535,
    -0.014300143522286599,
    0.07904677225341658,
    -0.11871804979426036,
    0.18648572519119352,
    3.499524284227276,
]
GAIA_STAR_DIST_MIN_MAG = 3.5
GAIA_STAR_DIST_MAX_MAG = 18
GAIA_STAR_DENS_PER_ARCMIN2 = 1.0358535145269943
GAIA_STAR_RAD_POLY = [1.36055007e-03, -1.55098040e-01,  3.46641671e+00]
BMASK_EXPAND_GAIA_STAR = 2**24

LOGGER = logging.getLogger(__name__)


def gen_gaia_mag_rad(*, rng, num, rad_dist):
    """Generate GAIA stars with radii.

    Parameters
    ----------
    rng : np.random.RandomState
        An RNG instance to use.
    num : int
        The number of stars to generate.
    rad_dist : str
        The distribution from which to draw the star radii. One of "gaia-des" or
        "uniform".

    Returns
    -------
    mag_g : array-like, shape (num,)
        The g band magnitude.
    rad : array-like, shape (num,)
        The mask radius for a DES-like survey.
    """
    u = rng.uniform(size=num)
    mag_g = np.polyval(GAIA_STAR_DIST_POLY, np.arcsinh(u/1e-6))
    if rad_dist == "gaia-des":
        ply = np.poly1d(GAIA_STAR_RAD_POLY)
        log10rad = ply(mag_g)
        rad = np.power(10, log10rad)
    else:
        rad = rng.uniform(size=num) * 20 + 5
    return mag_g, rad


def gen_stars(
    *, rng, pos_bounds, coadd_wcs, dens_factor, rad_dist,
    interp=None, mask_expand_rad=None, apodize=None
):
    """Generate GAIA stars for a simulation.

    Parameters
    ----------
    rng : np.random.RandomState
        An RNG instance to use.
    pos_bounds : 2-tuple of floats
        The range in which to generate u and v.
    coadd_wcs : a non-celestial galsim WCS object
        The WCS for the coadd.
    dens_factor : float
        The factor by which to adjust the star density. A value of 1.0 results in a
        density of roughly 1 star per arcmin^2.
    rad_dist : str
        The distribution from which to draw the star radii. One of "gaia-des" or
        "uniform".
    interp : ignored by this function
    mask_expand_rad : ignored by this function
    apodize : ignored by this function

    Returns
    -------
    stars : structured array
        A structured array with colums x, y and radius_pixels giving the location
        of the star and its mask radius in pixels.
    """
    dt = [
        ("x", "f8"),
        ("y", "f8"),
        ("radius_pixels", "f8")
    ]
    area = (pos_bounds[1] - pos_bounds[0])/60
    area = area**2
    num = rng.poisson(lam=area*GAIA_STAR_DENS_PER_ARCMIN2 * dens_factor)
    LOGGER.debug("generating %d stars", num)
    if num > 0:
        _, rad = gen_gaia_mag_rad(rng=rng, num=num, rad_dist=rad_dist)
        upos = rng.uniform(low=pos_bounds[0], high=pos_bounds[1], size=num)
        vpos = rng.uniform(low=pos_bounds[0], high=pos_bounds[1], size=num)
        x, y = coadd_wcs.toImage(upos, vpos)
        return np.array(list(zip(x, y, rad)), dtype=dt)
    else:
        return np.empty(0, dtype=dt)


def mask_stars(*, rng, mbobs, stars, interp_cfg, apodize_cfg, mask_expand_rad):
    """Mask according to a GAIA star catalog.

    Parameters
    ----------
    rng : np.random.RandomState
        An RNG instance to use.
    mbobs : ngmix.MultiBandObsList
        The coadd data in mbobs form.
    stars : array-like
        The star data to build the mask from.
    interp_cfg : dict
        A dictionary of keywords to pass to the interpolation function.
    apodize_cfg : dict
        A dictionary of keywords to use if apodizing the mask.
    mask_expand_rad : float
        The number of pixels to expand the bit mask by.
    """
    if len(stars) > 0:
        method = None
        if not apodize_cfg["skip"]:
            method = "apodize"

        if not interp_cfg["skip"]:
            if method is not None:
                raise RuntimeError("Only one interpolation method can be set!")

            if interp_cfg["fill_isolated_with_noise"]:
                method = "interp-noise"
            else:
                method = "interp"

        if method is not None:
            apply_foreground_masking_corrections(
                mbobs=mbobs,
                xm=stars["x"],
                ym=stars["y"],
                rm=stars["radius_pixels"],
                method=method,
                mask_expand_rad=mask_expand_rad,
                mask_bit_val=BMASK_GAIA_STAR,
                expand_mask_bit_val=BMASK_EXPAND_GAIA_STAR,
                interp_bit_val=BMASK_SPLINE_INTERP,
                symmetrize=False,
                ap_rad=apodize_cfg["ap_rad"],
                iso_buff=interp_cfg['iso_buff'],
                rng=rng,
            )

            for i in range(len(mbobs)):
                msk = (
                    ((mbobs[i][0].bmask & BMASK_GAIA_STAR) != 0)
                    |
                    ((mbobs[i][0].bmask & BMASK_EXPAND_GAIA_STAR) != 0)
                )
                if np.all(msk):
                    raise RuntimeError("All pixels are masked by the star!")
