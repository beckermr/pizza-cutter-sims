import numpy as np

import ngmix
from metadetect.metadetect import do_metadetect


def run_metadetect(*, rng, config, wcs, image, bmask, ormask, noise, psf, weight):
    """Run metadetect on an input sim.

    Parameters
    ----------
    rng : np.random.RandomState
        An RNG to use for running metadetection.
    config : dict
        A dictionary with the metadetection config information.
    wcs : galsim.BaseWCS or similar
        The coadd WCS object representing the WCS of the final images.
    image : np.ndarray
        The coadded image
    bmask : np.ndarray
        The bit mask for the coadded image.
    ormask : np.ndarray
        The logical "OR" mask for the coadded image.
    noise : np.ndarray
        A noise image for the coadd.
    psf : np.ndarray
        The coadd PSF image.
    weight : np.ndarray
        The weight map for the coadd.

    Returns
    -------
    res : dict
        Dictioanry with metadetection results.
    """
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
        psf,
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
        image,
        weight=weight,
        bmask=bmask,
        ormask=ormask,
        jacobian=im_jac,
        psf=psf_obs,
        noise=noise,
    )

    mbobs = ngmix.MultiBandObsList()
    obslist = ngmix.ObsList()
    obslist.append(obs)
    mbobs.append(obslist)

    return do_metadetect(config, mbobs, rng)
