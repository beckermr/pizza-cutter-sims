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
BMASK_EXPAND_GAIA_STAR = 2**24


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
    coadd_wcs : a non-celestial galsim WCS object
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
        x, y = coadd_wcs.toImage(upos, vpos)
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

            cdata["mfrac"][msk] = 1.0
            cdata["ormask"][msk] |= SIM_BMASK_GAIA_STAR
            cdata["bmask"][msk] |= BMASK_SPLINE_INTERP
            cdata["bmask"][msk] |= BMASK_GAIA_STAR

            if not interp_cfg.get('skip', False):
                cdata["image"][msk] = np.nan
                cdata["weight"][msk] = 0.0
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
                    or (
                        "extra_noises" in cdata
                        and any(im is None for im in interp_noises)
                    )
                ):
                    raise RuntimeError("Interpolation for gaia stars returned None!")
                else:
                    cdata["image"] = interp_image
                    cdata["noise"] = interp_noise
                    if "extra_noises" in cdata:
                        cdata["extra_noises"] = interp_noises
        elif np.all(msk):
            raise RuntimeError("All pixels are masked by the star!")


@njit
def _intersects(row, col, radius_pixels, nrows, ncols):
    """
    low level routine to check if the mask intersects the image.
    For simplicty just check the bounding rectangle

    Parameters
    ----------
    row, col: float
        The row and column of the mask center
    radius_pixels: float
        The radius for the star mask
    nrows, ncols: int
        Shape of the image

    Returns
    -------
    True if it intersects, otherwise False
    """

    low_row = -radius_pixels
    high_row = nrows + radius_pixels - 1
    low_col = -radius_pixels
    high_col = ncols + radius_pixels - 1

    if (
        row > low_row and row < high_row and
        col > low_col and col < high_col
    ):
        return True
    else:
        return False


@njit
def _build_mask_image(rows, cols, radius_pixels, mask):
    nrows, ncols = mask.shape
    nmasked = 0

    for istar in range(rows.size):
        row = rows[istar]
        col = cols[istar]

        rad = radius_pixels[istar]
        rad2 = rad * rad

        if not _intersects(row, col, rad, nrows, ncols):
            continue

        for irow in range(nrows):
            rowdiff2 = (row - irow)**2
            for icol in range(ncols):

                r2 = rowdiff2 + (col - icol)**2
                if r2 < rad2:
                    mask[irow, icol] = 1
                    nmasked += 1

    return nmasked


def build_mask_image(rows, cols, radius_pixels, dims, symmetrize=False):
    """Build image of 0 or 1 for masks.

    Parameters
    ----------
    rows, cols: arrays
        Arrays of rows/cols of the mask hole locations in the "local" coordimnates
        of the image.These positions may be off the image.
    radius_pixels: array
        The radius for each mask hole.
    dims: tuple of ints
        The shape of the mask.
    symmetrize: bool, optional
        If True, will symmetrize the mask with a 90 degree rotation. Default is False.

    Returns
    -------
    mask: array
        The array to set to 1 if something is masked, 0 otherwise.
    """

    # must be native byte order for numba
    cols = cols.astype('f8')
    rows = rows.astype('f8')
    radius_pixels = radius_pixels.astype('f8')

    mask = np.zeros(dims, dtype='i4')

    _build_mask_image(rows, cols, radius_pixels, mask)

    if symmetrize:
        if mask.shape[0] != mask.shape[1]:
            raise ValueError("Only square images can be symmetrized!")
        mask |= np.rot90(mask)

    return mask


def apply_mask_mbobs(mbobs, mask_catalog, maskflags, symmetrize=False):
    """Expands masks in an mbobs. This will expand the masks by setting the
    weight to zero, mfrac to 1, and setting maskflags in the bmask for every obs.

    Parameters
    ----------
    mbobs: ngmix.MultiBandObsList
        The observations to modify.
    mask_catalog: np.ndarray, optional
        If not None, this array should have columns 'x', 'y' and 'radius_pixels'
        that gives the location and radius of any masked objects in the image.
        Default of None indicates no catalog.
    maskflags: int
        The bit to set in the bit mask.
    symmetrize: bool, optional
        If True, will symmetrize the mask with a 90 degree rotation. Default is False.
    """
    dims = None
    for obslist in mbobs:
        for obs in obslist:
            dims = obs.image.shape
            break
    if dims is None:
        raise RuntimeError("Cannot expand the masks on an empty observation!")

    mask = build_mask_image(
        mask_catalog['y'],
        mask_catalog['x'],
        mask_catalog['radius_pixels'],
        dims,
        symmetrize=symmetrize,
    )
    msk = mask != 0
    if np.any(msk):
        for obslist in mbobs:
            for obs in obslist:
                with obs.writeable():
                    if hasattr(obs, "mfrac"):
                        obs.mfrac[msk] = 1
                    obs.weight[msk] = 0
                    obs.bmask[msk] |= maskflags


def apply_mask_mfrac(mfrac, mask_catalog, symmetrize=False):
    """Expand masks in an mfrac image. This will set mfrac to 1 in the expanded
    mask region.

    Parameters
    ----------
    mfrac: array
        The masked fraction image to modify.
    mask_catalog: np.ndarray, optional
        If not None, this array should have columns 'x', 'y' and 'radius_pixels'
        that gives the location and radius of any masked objects in the image.
        Default of None indicates no catalog.
    symmetrize: bool, optional
        If True, will symmetrize the mask with a 90 degree rotation. Default is False.
    """
    mask = build_mask_image(
        mask_catalog['y'],
        mask_catalog['x'],
        mask_catalog['radius_pixels'],
        mfrac.shape,
        symmetrize=symmetrize,
    )
    msk = mask != 0
    if np.any(msk):
        mfrac[msk] = 1


def apply_mask_bit_mask(bmask, mask_catalog, maskflags, symmetrize=False):
    """Expand masks in a bit mask. This will set maskflags in the expanded
    mask region.

    Parameters
    ----------
    bmask: array
        The git mask to modify.
    mask_catalog: np.ndarray, optional
        If not None, this array should have columns 'x', 'y' and 'radius_pixels'
        that gives the location and radius of any masked objects in the image.
        Default of None indicates no catalog.
    maskflags: int
        The bit to set in the bit mask.
    symmetrize: bool, optional
        If True, will symmetrize the mask with a 90 degree rotation. Default is False.
    """
    mask = build_mask_image(
        mask_catalog['y'],
        mask_catalog['x'],
        mask_catalog['radius_pixels'],
        bmask.shape,
        symmetrize=symmetrize,
    )
    msk = mask != 0
    if np.any(msk):
        bmask[msk] |= maskflags
