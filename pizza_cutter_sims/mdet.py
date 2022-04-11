import copy
import numpy as np

import ngmix
from metadetect.metadetect import do_metadetect
from pizza_cutter_sims.stars import (
    BMASK_GAIA_STAR,
    BMASK_EXPAND_GAIA_STAR,
)


def make_mbobs_from_coadd_data(
    *, wcs, cdata_list,
):
    """Make an ngmix.MultiBandObsList from the coadd data.

    Parameters
    ----------
    wcs : galsim.BaseWCS or similar
        The coadd WCS object representing the WCS of the final images.
    cdata_list : list of dict
        The dictionary of coadd data from
        `pizza_cutter_sims.pizza_cutter.run_des_pizza_cutter_coadding_on_sim`.
        One obs will be made per entry in the list and inserted as a band in the final
        mbobs.

    Returns
    -------
    mbobs : ngmix.MultiBandObsList
        The coadd data in mbobs form.
    """
    mbobs = ngmix.MultiBandObsList()

    for cdata in cdata_list:
        image = cdata["image"]
        bmask = cdata["bmask"]
        ormask = cdata["ormask"]
        noise = cdata["noise"]
        psf = cdata["psf"]
        weight = cdata["weight"]
        mfrac = cdata["mfrac"]

        psf_cen = (psf.shape[0] - 1)/2
        im_cen = (image.shape[0] - 1)/2

        # make the mbobs
        psf_jac = ngmix.jacobian.Jacobian(
            x=psf_cen,
            y=psf_cen,
            dudx=wcs.dudx,
            dudy=wcs.dudy,
            dvdx=wcs.dvdx,
            dvdy=wcs.dvdy,
        )
        target_s2n = 500.0
        target_noise = np.sqrt(np.sum(psf ** 2)) / target_s2n
        psf_obs = ngmix.Observation(
            psf.copy(),
            weight=np.ones_like(psf)/target_noise**2,
            jacobian=psf_jac,
        )

        im_jac = ngmix.jacobian.Jacobian(
            x=im_cen,
            y=im_cen,
            dudx=wcs.dudx,
            dudy=wcs.dudy,
            dvdx=wcs.dvdx,
            dvdy=wcs.dvdy,
        )
        obs = ngmix.Observation(
            image.copy(),
            weight=weight.copy(),
            bmask=bmask.copy(),
            ormask=ormask.copy(),
            jacobian=im_jac,
            psf=psf_obs,
            noise=noise.copy(),
            mfrac=np.clip(mfrac.copy(), 0, 1),
        )

        obslist = ngmix.ObsList()
        obslist.append(obs)
        mbobs.append(obslist)

    return mbobs


def gen_metadetect_color_dep(
    *,
    psfs,
    coadd_wcs,
    mbobs,
    coadd_cen_pos,
    color_range,
    ncolors,
    flux_zeropoints,
):
    """
    Generate the inputs needed by metadetect for handling color dependent PSFs.

    Parameters
    ----------
    psfs : list of PSF models
        The PSF models per band. They are objects that have a getPSF attribute.
    coadd_wcs : galsim WCS
        The galsim WCS object for the coadd.
    mbobs : ngmix.MultiBandObsList
        The multiband obs list for the obs. Copied and PSFs are replaced.
    coadd_cen_pos : galsim.PositionD
        The position of the center of the coadd image.
    color_range : list of floats
        the range of colors over which to make the models.
    ncolors : int
        The number of different models to make.
    flux_zeropoints : list of float
        The flux zeropoints to compute magnitudes in each band.

    Returns
    -------
    color_key_func: function
        If given, a function that computes a color or tuple of colors to key the
        `color_dep_mbobs` dictionary given an input set of fluxes from the mbobs.
    color_dep_mbobs: dict of mbobs
        A dictionary of color-dependently rendered observations of the mbobs for use
        in color-dependent metadetect.
    """
    colors = np.linspace(color_range[0], color_range[1], ncolors)

    def color_key_func(fluxes):
        if np.any(~np.isfinite(fluxes)):
            return None

        if fluxes[0] < 0 or fluxes[1] < 0:
            if fluxes[0] < fluxes[1]:
                return ncolors - 1
            else:
                return 0
        else:
            mag0 = flux_zeropoints[0] - np.log10(fluxes[0])/0.4
            mag1 = flux_zeropoints[1] - np.log10(fluxes[1])/0.4
            color = mag0 - mag1

            if color <= color_range[0]:
                return 0
            elif color >= color_range[1]:
                return ncolors - 1
            else:
                dcolors = colors[1] - colors[0]
                return int((color - color_range[0])/dcolors + 0.5)

    wcs = coadd_wcs.jacobian(image_pos=coadd_cen_pos)
    psf_dim = 53
    psf_cen = (psf_dim-1)/2
    psf_jac = ngmix.jacobian.Jacobian(
        x=psf_cen,
        y=psf_cen,
        dudx=wcs.dudx,
        dudy=wcs.dudy,
        dvdx=wcs.dvdx,
        dvdy=wcs.dvdy,
    )
    target_s2n = 500.0

    color_dep_mbobs = {}
    for cind, color in enumerate(colors):
        _mbobs = mbobs.copy()
        for pind, psf in enumerate(psfs):
            psf_im = psf.getPSF(coadd_cen_pos, color=color).drawImage(
                nx=psf_dim, ny=psf_dim,
                wcs=wcs,
            ).array

            target_noise = np.sqrt(np.sum(psf_im ** 2)) / target_s2n
            psf_obs = ngmix.Observation(
                psf_im,
                weight=np.ones_like(psf_im)/target_noise**2,
                jacobian=psf_jac,
            )
            _mbobs[pind][0].psf = psf_obs

        color_dep_mbobs[cind] = _mbobs

    return color_key_func, color_dep_mbobs


def run_metadetect(
    *, rng, config, mbobs, color_key_func=None, color_dep_mbobs=None,
):
    """Run metadetect on an input sim.

    Parameters
    ----------
    rng : np.random.RandomState
        An RNG to use for running metadetection.
    config : dict
        A dictionary with the metadetection config information.
    mbobs : ngmix.MultiBandObsList
        The coadd data in mbobs form.
    color_key_func: function, optional
        If given, a function that computes a color or tuple of colors to key the
        `color_dep_mbobs` dictionary given an input set of fluxes from the mbobs.
    color_dep_mbobs: dict of mbobs, optional
        A dictionary of color-dependently rendered observations of the mbobs for use
        in color-dependent metadetect.

    Returns
    -------
    res : dict
        Dictioanry with metadetection results.
    """
    _cfg = copy.deepcopy(config)
    _cfg.pop("color_dep_psf", None)
    if color_key_func is not None and color_dep_mbobs is not None:
        mdet_res = do_metadetect(
            _cfg, mbobs, rng,
            color_key_func=color_key_func, color_dep_mbobs=color_dep_mbobs,
        )
    else:
        mdet_res = do_metadetect(_cfg, mbobs, rng)

    if mdet_res is not None:
        new_mdet_res = {}
        for k, v in mdet_res.items():
            if v is not None:
                msk = (
                    ((v['bmask'] & BMASK_EXPAND_GAIA_STAR) == 0)
                    & ((v['bmask'] & BMASK_GAIA_STAR) == 0)
                )
                if np.any(msk):
                    new_mdet_res[k] = v[msk]
                else:
                    new_mdet_res[k] = None
        mdet_res = new_mdet_res

    return mdet_res
