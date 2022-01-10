import logging
import copy
import tempfile
import numpy as np
import galsim

from pizza_cutter_sims.mdet import run_metadetect, make_mbobs_from_coadd_data
from pizza_cutter_sims.pizza_cutter import run_des_pizza_cutter_coadding_on_sim
from pizza_cutter_sims.sim import generate_sim
from pizza_cutter_sims.stars import mask_stars

LOGGER = logging.getLogger(__name__)


def run_end2end_pair_with_shear(
    *, rng_seed, gal_rng_seed, star_rng_seed,
    coadd_rng_seed, mdet_rng_seed, cfg, g1, g2, swap12,
):
    """Run a pair of simulations with opposite shears.

    By default, the shear on the 1-axis is set to +g1 and -g1 for the pair of sims.

    Parameters
    ----------
    rng_seed : int
        The seed for the sim RNG.
    gal_rng_seed : int
        The seed for the gal RNG.
    star_rng_seed : int
        The seed for the star RNG.
    coadd_rng_seed : int
        The seed for the coadding code RNG.
    mdet_rng_seed : int
        The seed for the metadetection RNG.
    cfg : dict
        The config dictionary for the sim.
    g1 : float
        The shear on the 1-axis.
    g2 : float
        The shear on the 2-axis.
    swap12 : bool
        If True, the roles of g1 and g2 are swapped.

    Returns
    -------
    pres : dict
        A dictionary with the metadetection results for the +g1 shear.
    mres : dict
        A dictionary with the metadetection results for the -g1 shear.
    """
    if swap12:
        g1p = g2
        g2p = g1
        g1m = g2
        g2m = -g1
    else:
        g1p = g1
        g2p = g2
        g1m = -g1
        g2m = g2

    try:
        pres = run_end2end_with_shear(
            rng_seed=rng_seed,
            gal_rng_seed=gal_rng_seed,
            star_rng_seed=star_rng_seed,
            coadd_rng_seed=coadd_rng_seed,
            mdet_rng_seed=mdet_rng_seed,
            cfg=copy.deepcopy(cfg),
            g1=g1p,
            g2=g2p,
        )

        mres = run_end2end_with_shear(
            rng_seed=rng_seed,
            gal_rng_seed=gal_rng_seed,
            star_rng_seed=star_rng_seed,
            coadd_rng_seed=coadd_rng_seed,
            mdet_rng_seed=mdet_rng_seed,
            cfg=copy.deepcopy(cfg),
            g1=g1m,
            g2=g2m,
        )

        return pres, mres
    except KeyError:
        raise
    except TypeError:
        raise
    except Exception as e:
        import traceback
        print("sim failed: %s\n%s" % (repr(e), traceback.format_exc()), flush=True)
        return None, None


def run_end2end_with_shear(
    *, rng_seed, gal_rng_seed, star_rng_seed,
    coadd_rng_seed, mdet_rng_seed, cfg, g1, g2
):
    """Run a full sim end-to-end w/ analysis.

    Parameters
    ----------
    rng_seed : int
        The seed for the sim RNG.
    gal_rng_seed : int
        The seed for the gal RNG.
    star_rng_seed : int
        The seed for the star RNG.
    coadd_rng_seed : int
        The seed for the coadding code RNG.
    mdet_rng_seed : int
        The seed for the metadetection RNG.
    cfg : dict
        The config dictionary for the sim.
    g1 : float
        The shear on the 1-axis.
    g2 : float
        The shear on the 2-axis.

    Returns
    -------
    res : dict
        A dictionary with the metadetection results.
    """
    cfg["shear"]["g1"] = g1
    cfg["shear"]["g2"] = g2

    LOGGER.info(
        "RNG seeds sim|gal|stars|coadd|mdet: %d|%d|%d|%d|%d",
        rng_seed,
        gal_rng_seed,
        star_rng_seed,
        coadd_rng_seed,
        mdet_rng_seed,
    )

    rng = np.random.RandomState(seed=rng_seed)
    gal_rng = np.random.RandomState(seed=gal_rng_seed)
    star_rng = np.random.RandomState(seed=star_rng_seed)
    sdata = generate_sim(
        rng=rng,
        gal_rng=gal_rng,
        star_rng=star_rng,
        coadd_config=cfg["coadd"],
        se_config=cfg["se"],
        psf_config=cfg["psf"],
        gal_config=cfg["gal"],
        star_config=cfg["star"],
        layout_config=cfg["layout"],
        msk_config=cfg["msk"],
        shear_config=cfg["shear"],
        skip_coadding=cfg["pizza_cutter"]["skip"],
    )

    coadd_rng = np.random.RandomState(seed=coadd_rng_seed)
    if not cfg["pizza_cutter"]["skip"]:
        with tempfile.TemporaryDirectory() as tmpdir:
            cdata = run_des_pizza_cutter_coadding_on_sim(
                rng=coadd_rng,
                tmpdir=tmpdir,
                single_epoch_config=cfg["pizza_cutter"]["single_epoch_config"],
                img=sdata["img"],
                wgt=sdata["wgt"],
                msk=sdata["msk"],
                bkg=sdata["bkg"],
                info=sdata["info"],
                n_extra_noise_images=0,
            )
    else:
        LOGGER.info("skipping coadding w/ pizza-cutter")
        coadd_cen = (sdata["img"][0].shape[0]-1)/2
        coadd_cen_pos = galsim.PositionD(x=coadd_cen, y=coadd_cen)
        psf = sdata["psfs"][0].getPSF(coadd_cen_pos).drawImage(
            nx=53, ny=53,
            wcs=sdata["coadd_wcs"].jacobian(coadd_cen_pos),
        ).array
        cdata = dict(
            image=sdata["img"][0].copy(),
            bmask=sdata["msk"][0].copy(),
            ormask=sdata["msk"][0].copy(),
            weight=sdata["wgt"][0].copy(),
            mfrac=np.zeros_like(sdata["img"][0]),
            noise=coadd_rng.normal(
                size=sdata["img"][0].shape,
                scale=np.sqrt(1.0/sdata["wgt"][0]),
            ),
            psf=psf,
        )

    mbobs = make_mbobs_from_coadd_data(
        wcs=sdata["coadd_wcs"],
        cdata=cdata,
    )

    mask_stars(
        rng=star_rng,
        mbobs=mbobs,
        stars=sdata["stars"],
        interp_cfg=cfg["star"]["interp"],
        apodize_cfg=cfg["star"]["apodize"],
        mask_expand_rad=cfg["star"]["mask_expand_rad"],
    )

    mdet_rng = np.random.RandomState(seed=mdet_rng_seed)
    return run_metadetect(
        rng=mdet_rng,
        config=cfg["metadetect"],
        mbobs=mbobs,
    )
