import numpy as np

import ngmix
from metadetect.metadetect import do_metadetect
from pizza_cutter_sims.stars import (
    BMASK_GAIA_STAR,
    BMASK_EXPAND_GAIA_STAR,
)


def make_mbobs_from_coadd_data(
    *, wcs, cdata,
):
    """Make an ngmix.MultiBandObsList from the coadd data.

    Parameters
    ----------
    wcs : galsim.BaseWCS or similar
        The coadd WCS object representing the WCS of the final images.
    cdata : dict
        The dictionary of coadd data from
        `pizza_cutter_sims.pizza_cutter.run_des_pizza_cutter_coadding_on_sim`

    Returns
    -------
    mbobs : ngmix.MultiBandObsList
        The coadd data in mbobs form.
    """
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

    mbobs = ngmix.MultiBandObsList()
    obslist = ngmix.ObsList()
    obslist.append(obs)
    mbobs.append(obslist)

    return mbobs


def run_metadetect(
    *, rng, config, mbobs,
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

    Returns
    -------
    res : dict
        Dictioanry with metadetection results.
    """
    mdet_res = do_metadetect(config, mbobs, rng)

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
