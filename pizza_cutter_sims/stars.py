import numpy as np
from numba import njit

from pizza_cutter.slice_utils.interpolate import interpolate_image_at_mask
from pizza_cutter.des_pizza_cutter import (
    BMASK_SPLINE_INTERP, BMASK_GAIA_STAR
)

from .constants import SIM_BMASK_GAIA_STAR

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


def gen_gaia_mag_rad(*, rng, num):
    """Generate GAIA stars with radii.

    Parameters
    ----------
    rng : np.random.RandomState
        An RNG instance to use.
    num : int
        The number of stars to generate.

    Returns
    -------
    mag_g : array-like, shape (num,)
        The g band magnitude.
    rad : array-like, shape (num,)
        The mask radius for a DEs-like survey.
    """
    u = rng.uniform(size=num)
    mag_g = np.polyval(GAIA_STAR_DIST_POLY, np.arcsinh(u/1e-6))
    ply = np.poly1d(GAIA_STAR_RAD_POLY)
    log10rad = ply(mag_g)
    rad = np.power(10, log10rad)
    return mag_g, rad


def gen_stars(
    *, rng, pos_bounds, coadd_wcs, dens_factor, interp=None, mask_expand_rad=None
):
    """Generate GAIA stars for a simulation.

    Parameters
    ----------
    rng : np.random.RandomState
        An RNG instance to use.
    pos_bounds : 2-tuple of floats
        The range in which to generate u and v.
    coadd_wcs : an AffineWCS object
        The WCS for the coadd.
    dens_factor : float
        The factor by which to adjust the star density. A value of 1.0 results in a
        density of roughly 1 star per arcmin^2.
    interp : ignored by this function
    mask_expand_rad : ignored by this function

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
    if num > 0:
        _, rad = gen_gaia_mag_rad(rng=rng, num=num)
        upos = rng.uniform(low=pos_bounds[0], high=pos_bounds[1], size=num)
        vpos = rng.uniform(low=pos_bounds[0], high=pos_bounds[1], size=num)
        x, y = coadd_wcs.sky2image(upos, vpos)
        return np.array(list(zip(x, y, rad)), dtype=dt)
    else:
        return np.empty(0, dtype=dt)


@njit
def _set_gaia_star_in_bmask(xs, ys, rads, bmask):
    ny, nx = bmask.shape
    ns = xs.shape[0]

    for i in range(ns):
        x = xs[i]
        y = ys[i]
        rad2 = rads[i]**2
        for _y in range(ny):
            dy2 = (_y - y)**2
            for _x in range(nx):
                dr2 = (_x - x)**2 + dy2
                if dr2 < rad2:
                    bmask[_y, _x] |= BMASK_GAIA_STAR


def mask_and_interp_stars(*, rng, cdata, stars, interp_cfg):
    """Mask and interpolate according to a GAIA star catalog.

    Parameters
    ----------
    rng : np.random.RandomState
        An RNG instance to use.
    cdata : dict
        A dictionary with the coadded data. The data in the dictionary will be
        modified.
    stars : array-like
        The star data to build the mask from.
    interp_cfg : dict
        A dictionary of keywords to pass to the interpolation function.
    """
    if len(stars) > 0:
        _set_gaia_star_in_bmask(
            stars["x"], stars["y"], stars["radius_pixels"], cdata["bmask"]
        )
        msk = (cdata["bmask"] & BMASK_GAIA_STAR) != 0
        if np.any(msk) and not np.all(msk):
            med_wgt = np.median(
                cdata["weight"][(cdata["bmask"] == 0) & (cdata["weight"] > 0)]
            )
            cdata["image"][msk] = np.nan
            cdata["weight"][msk] = 0.0
            cdata["mfrac"][msk] = 1.0
            cdata["ormask"][msk] |= SIM_BMASK_GAIA_STAR
            cdata["bmask"][msk] |= BMASK_SPLINE_INTERP
            cdata["bmask"][msk] |= BMASK_GAIA_STAR

            interp_image = interpolate_image_at_mask(
                image=cdata["image"], bad_msk=msk, maxfrac=1.0,
                rng=rng, weight=med_wgt, **interp_cfg,
            )
            interp_noise = interpolate_image_at_mask(
                image=cdata["noise"], bad_msk=msk, maxfrac=1.0,
                rng=rng, weight=med_wgt, **interp_cfg,
            )
            if "extra_noises" in cdata:
                interp_noises = []
                for nse in cdata["extra_noises"]:
                    interp_noises.append(interpolate_image_at_mask(
                        image=nse, bad_msk=msk, maxfrac=1.0,
                        rng=rng, weight=med_wgt, **interp_cfg,
                    ))
            if (
                interp_image is None
                or interp_noise is None
                or ("extra_noises" in cdata and any(im is None for im in interp_noises))
            ):
                raise RuntimeError("Interpolation for gaia stars returned None!")
            else:
                cdata["image"] = interp_image
                cdata["noise"] = interp_noise
                if "extra_noises" in cdata:
                    cdata["extra_noises"] = interp_noises
        elif np.all(msk):
            raise RuntimeError("All pixels are masked by the star!")
