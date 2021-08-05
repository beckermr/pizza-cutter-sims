import numpy as np
import galsim

from pizza_cutter.des_pizza_cutter._affine_wcs import AffineWCS

from .wcs import gen_affine_wcs
from .constants import (
    SIM_BMASK_BADCOLS,
    SIM_BMASK_COSMICS,
    SIM_BMASK_STREAKS,
    MAGZP_REF,
)
from .psf import gen_psf
from .gals import gen_gals
from .masking import (
    generate_bad_columns,
    generate_cosmic_rays,
    generate_streaks,
)
from .stars import gen_stars


def generate_sim(
    *,
    rng,
    gal_rng,
    star_rng,
    coadd_config,
    se_config,
    psf_config,
    gal_config,
    star_config,
    layout_config,
    msk_config,
    shear_config,
    skip_coadding,
):
    """Generate a set of SE images and their metadata for coadding.

    Parameters
    ----------
    rng : np.random.RandomState
        An RNG instance to use to build info for this band.
    gal_rng : np.random.RandomState
        An RNG instance to use to build galaxies.
    star_rng : np.random.RandomState
        An RNG instance to use to build stars.
    coadd_config : dict
        A dictionary with info about the coadd image.
    se_config : dict
        A dictionary with info about the SE images.
    psf_config : dict
        A dictionary with info about the PSF.
    gal_config : dict
        A dictionary with info about the galaxies.
    star_config : dict
        A dictionatu witjh info about the stars.
    layout_config : dict
        A dictionary with info about the galaxy layout.
    msk_config : dict
        A dictionary with info about masking effects being applied.
    shear_config : dict
        A dictionary with info about the true shear applied.
    skip_coadding : bool
        If True, skip coadding. Requires the SE image to have
        the same WCS as the coadd image and that there be only 1 SE image.

    Returns
    -------
    data : dict
        A dictionary with the follow keys and values:

        info : dict
            A dictionary with the information about the images in the
            correct format for coadding.
        img : list of np.ndarray
            The images to coadd.
        wgt : list of np.ndarray
            The weight maps of the imags to coadd.
        msk : list of np.ndarray
            The bit masks of the images to coadd.
        bkg : list of np.ndarray
            The background images.
    """
    info = {}
    src_info = [dict() for _ in range(se_config["n_images"])]
    info['src_info'] = src_info
    images = []
    weights = []
    bmasks = []
    bkgs = []
    psfs = []
    wcss = []

    # first build the WCS and image data
    coadd_image_shape = coadd_config["central_size"] + coadd_config["buffer_size"] * 2
    assert coadd_image_shape % 2 == 1
    coadd_image_cen = (coadd_image_shape - 1) // 2
    info["affine_wcs_config"] = {
        "dudx": coadd_config["scale"],
        "dudy": 0,
        "dvdx": 0,
        "dvdy": coadd_config["scale"],
        "x0": coadd_image_cen,
        "y0": coadd_image_cen,
    }
    coadd_wcs = AffineWCS(**info["affine_wcs_config"])
    info["image_shape"] = [coadd_image_shape, coadd_image_shape]
    info["magzp"] = MAGZP_REF
    info["scale"] = 1.0
    info["position_offset"] = 0

    if not skip_coadding:
        se_factor = 1.7 * (
            1.0
            + se_config["wcs_config"]["scale_frac_std"]
            + np.sqrt(2) * se_config["wcs_config"]["shear_std"]
        )
        se_image_shape = int(se_factor * coadd_image_shape)
        if se_image_shape % 2 == 0:
            se_image_shape += 1
        se_image_cen = (se_image_shape - 1) // 2
    else:
        se_image_shape = coadd_image_shape
        se_image_cen = coadd_image_cen

    for ii in src_info:
        ii["magzp"] = MAGZP_REF
        ii["scale"] = 1.0
        ii["position_offset"] = 0

        # done here to make sure the RNG calls are consistent
        # may not be used below
        _wcs = gen_affine_wcs(
            rng=rng,
            world_origin=galsim.PositionD(x=0, y=0),
            origin=galsim.PositionD(x=se_image_cen, y=se_image_cen),
            **se_config["wcs_config"],
        )
        # use the coadd WCS if we are skipping coadding since they have to be
        # the same in this case
        if skip_coadding:
            _wcs = coadd_wcs

        wcss.append(_wcs)
        ii['affine_wcs_config'] = {
            'dudx': float(_wcs.dudx),
            'dudy': float(_wcs.dudy),
            'dvdx': float(_wcs.dvdx),
            'dvdy': float(_wcs.dvdy),
            'x0': float(_wcs.origin.x),
            'y0': float(_wcs.origin.y),
        }
        ii['image_shape'] = [se_image_shape, se_image_shape]
        ii['image_flags'] = 0

    # now make PSFs
    for ii in src_info:
        _psf_config, _psf_obj = gen_psf(
            rng=rng,
            psf_config=psf_config,
        )
        ii['galsim_psf_config'] = _psf_config
        psfs.append(_psf_obj)

    # generate galaxies and stars
    pos_bounds = (
        -0.5 * coadd_config["central_size"] * coadd_config["scale"],
        0.5 * coadd_config["central_size"] * coadd_config["scale"],
    )
    gals, upos, vpos, img_noise = gen_gals(
        rng=gal_rng,
        layout_config=layout_config,
        gal_config=gal_config,
        pos_bounds=pos_bounds,
    )

    stars = gen_stars(
        rng=star_rng,
        pos_bounds=pos_bounds,
        coadd_wcs=coadd_wcs,
        **star_config
    )

    # now render and apply masks
    g1true = shear_config["g1"]
    g2true = shear_config["g2"]
    bnds = galsim.BoundsI(
        0, se_image_shape-1,
        0, se_image_shape-1,
    )
    for se_ind in range(len(src_info)):
        _psf = psfs[se_ind]
        _wcs = wcss[se_ind]

        image = galsim.ImageD(bnds, dtype=np.float32, init_value=0)

        for gal, u, v in zip(gals, upos, vpos):
            if shear_config["scene"]:
                gal = gal.shift(u, v).shear(g1=g1true, g2=g2true)
            else:
                gal = gal.shear(g1=g1true, g2=g2true).shift(u, v)
            x, y = _wcs.uvToxy(u, v)
            galsim.Convolve(
                gal, _psf.getPSF(galsim.PositionD(x=x, y=y))
            ).drawImage(
                image=image,
                add_to_image=True,
                center=_wcs.origin,
                wcs=_wcs,
            )
        image = image.array

        bkg = np.zeros_like(image)
        weight = np.zeros_like(image)
        weight[:, :] = 1.0 / img_noise / img_noise
        image += (rng.normal(size=image.shape) * img_noise)
        image += bkg

        msk = np.zeros(image.shape, dtype=np.int32)
        # do not move these calls, keeps the options doing the same thing
        # when one or the other is turned off
        bad_col_rng = np.random.RandomState(rng.randint(1, 2**29))
        cray_rng = np.random.RandomState(rng.randint(1, 2**29))
        streak_rng = np.random.RandomState(rng.randint(1, 2**29))

        if msk_config["bad_columns"] or msk_config["bad_columns"] == {}:
            if msk_config["bad_columns"] is True:
                msk_config["bad_columns"] = {}

            _msk = generate_bad_columns(
                shape=(se_image_shape, se_image_shape),
                rng=bad_col_rng,
                **msk_config["bad_columns"],
            )
            msk[_msk] |= SIM_BMASK_BADCOLS
            image[_msk] = np.nan

        if msk_config["cosmic_rays"] or msk_config["cosmic_rays"] == {}:
            if msk_config["cosmic_rays"] is True:
                msk_config["cosmic_rays"] = {}

            _msk = generate_cosmic_rays(
                shape=(se_image_shape, se_image_shape),
                rng=cray_rng,
                **msk_config["cosmic_rays"],
            )
            msk[_msk] |= SIM_BMASK_COSMICS
            image[_msk] = np.nan

        if msk_config["streaks"] or msk_config["streaks"] == {}:
            if msk_config["streaks"] is True:
                msk_config["streaks"] = {}

            _msk = generate_streaks(
                shape=(se_image_shape, se_image_shape),
                rng=streak_rng,
                **msk_config["streaks"],
            )
            msk[_msk] |= SIM_BMASK_STREAKS
            image[_msk] = np.nan

        images.append(image)
        weights.append(weight)
        bmasks.append(msk)
        bkgs.append(bkg)

    # tuck the images in the info structure for the coadding
    for ind, ii in enumerate(info["src_info"]):
        # the epoch keys are set to make sure the pizza cutter slices cache
        # properly
        ii["image_path"] = images[ind]
        ii["image_ext"] = "epoch%d" % ind
        ii["bkg_path"] = bkgs[ind]
        ii["bkg_ext"] = "epoch%d" % ind
        ii["weight_path"] = weights[ind]
        ii["weight_ext"] = "epoch%d" % ind
        ii["bmask_path"] = bmasks[ind]
        ii["bmask_ext"] = "epoch%d" % ind
        ii["path"] = "epoch%d" % ind
        ii["filename"] = "epoch%d" % ind

    # here to be comaptible with pizza cutter
    info["path"] = "coadd"
    info["filename"] = "coadd"

    return {
        "info": info,
        "img": images,
        "wgt": weights,
        "msk": bmasks,
        "bkg": bkgs,
        "stars": stars,
    }
