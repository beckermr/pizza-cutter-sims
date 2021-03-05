import copy
import tempfile
import numpy as np

from pizza_cutter_sims.mdet import run_metadetect
from pizza_cutter_sims.pizza_cutter import run_des_pizza_cutter_coadding_on_sim
from pizza_cutter_sims.sim import generate_sim


def run_end2end_pair_with_shear(
    *, rng_seed, gal_rng_seed, coadd_rng_seed, mdet_rng_seed, cfg, g1, g2, swap12,
):
    """Run a pair of simulations with opposite shears.

    By default, the shear on the 1-axis is set to +g1 and -g1 for the pair of sims.

    Parameters
    ----------
    rng_seed : int
        The seed for the sim RNG.
    gal_rng_seed : int
        The seed for the gal RNG.
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

    pres = run_end2end_with_shear(
        rng_seed=rng_seed,
        gal_rng_seed=gal_rng_seed,
        coadd_rng_seed=coadd_rng_seed,
        mdet_rng_seed=mdet_rng_seed,
        cfg=copy.deepcopy(cfg),
        g1=g1p,
        g2=g2p,
    )

    mres = run_end2end_with_shear(
        rng_seed=rng_seed,
        gal_rng_seed=gal_rng_seed,
        coadd_rng_seed=coadd_rng_seed,
        mdet_rng_seed=mdet_rng_seed,
        cfg=copy.deepcopy(cfg),
        g1=g1m,
        g2=g2m,
    )

    return pres, mres


def run_end2end_with_shear(
    *, rng_seed, gal_rng_seed, coadd_rng_seed, mdet_rng_seed, cfg, g1, g2
):
    """Run a full sim end-to-end w/ analysis.

    Parameters
    ----------
    rng_seed : int
        The seed for the sim RNG.
    gal_rng_seed : int
        The seed for the gal RNG.
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

    rng = np.random.RandomState(seed=rng_seed)
    gal_rng = np.random.RandomState(seed=gal_rng_seed)
    sdata = generate_sim(
        rng=rng,
        gal_rng=gal_rng,
        coadd_config=cfg["coadd"],
        se_config=cfg["se"],
        psf_config=cfg["psf"],
        gal_config=cfg["gal"],
        layout_config=cfg["layout"],
        msk_config=cfg["msk"],
        shear_config=cfg["shear"],
    )

    coadd_rng = np.random.RandomState(seed=coadd_rng_seed)
    with tempfile.TemporaryDirectory() as tmpdir:
        cdata = run_des_pizza_cutter_coadding_on_sim(
            rng=coadd_rng,
            tmpdir=tmpdir,
            single_epoch_config=cfg["pizza_cutter"]["single_epoch_config"],
            img=sdata["img"],
            wgt=sdata["wgt"],
            msk=sdata["msk"],
            bkg=sdata["bkg"],
        )

    mdet_rng = np.random.RandomState(seed=mdet_rng_seed)
    return run_metadetect(
        rng=mdet_rng,
        config=cfg["metadetect"],
        wcs=sdata["info"]["affine_wcs"],
        image=cdata["image"],
        bmask=cdata["bmask"],
        ormask=cdata["ormask"],
        noise=cdata["noise"],
        psf=cdata["psf"],
        weight=cdata["weight"],
    )
