import numpy as np
import galsim

from .wcs import gen_affine_wcs
from .constants import SIM_BMASK_BADCOLS, SIM_BMASK_COSMICS
from .psf import gen_psf
from .gals import gen_gals
from .masking import generate_bad_columns, generate_cosmic_rays


def generate_sim(
    *,
    rng,
    gal_rng,
    coadd_config,
    se_config,
    psf_config,
    gal_config,
    layout_config,
    msk_config,
    shear_config,
):
    """Generate a set of SE images and their metadata for coadding.

    Parameters
    ----------
    rng : np.random.RandomState
        An RNG instance to use to build info for this band.
    gal_rng : np.random.RandomState
        An RNG instance to use to build galaxies.
    coadd_config : dict
        A dictionary with info about the coadd image.
    se_config : dict
        A dictionary with info about the SE images.
    psf_config : dict
        A dictionary with info about the PSF.
    gal_config : dict
        A dictionary with info about the galaxies.
    layout_config : dict
        A dictionary with info about the galaxy layout.
    msk_config : dict
        A dictionary with info about masking effects being applied.
    shear_config : dict
        A dictionary with info about the true shear applied.

    Returns
    -------
    data : dict
        A dictionary with the follow keys and values:

        info : dict
            A dictionary with the information about the images in the
            correct format for coadding.
        img : list of np.ndarray
            The images to coadd.
        wwgt : list of np.ndarray
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
    info["image_shape"] = [coadd_image_shape, coadd_image_shape]
    info["position_offset"] = 0

    se_image_shape = int(np.sqrt(2) * coadd_image_shape)
    if se_image_shape % 2 == 0:
        se_image_shape += 1
    se_image_cen = (se_image_shape - 1) // 2

    for ii in src_info:
        ii["scale"] = 1.0
        ii["position_offset"] = 0
        _wcs = gen_affine_wcs(
            rng=rng,
            world_origin=galsim.PositionD(x=0, y=0),
            origin=galsim.PositionD(x=se_image_cen, y=se_image_cen),
            **se_config["wcs_config"],
        )
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

    # generate galaxies
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

        image = galsim.ImageD(bnds, dtype=np.float32, init_value=0, wcs=_wcs)

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
            )
        image = image.array

        bkg = np.zeros_like(image)
        weight = np.zeros_like(image)
        weight[:, :] = 1.0 / img_noise / img_noise
        image += (rng.normal(size=image.shape) * img_noise)
        image += bkg

        msk = np.zeros(image.shape, dtype=np.int32)

        if msk_config["bad_columns"] or msk_config["bad_columns"] == {}:
            if msk_config["bad_columns"] is True:
                msk_config["bad_columns"] = {}
            _msk = generate_bad_columns(
                shape=(se_image_shape, se_image_shape),
                rng=rng,
                **msk_config["bad_columns"],
            )
            msk[_msk] |= SIM_BMASK_BADCOLS
            image[_msk] = np.nan

        if msk_config["cosmic_rays"] or msk_config["cosmic_rays"] == {}:
            if msk_config["cosmic_rays"] is True:
                msk_config["cosmic_rays"] = {}

            _msk = generate_cosmic_rays(
                shape=(se_image_shape, se_image_shape),
                rng=rng,
                **msk_config["cosmic_rays"],
            )
            msk[_msk] |= SIM_BMASK_COSMICS
            image[_msk] = np.nan

        images.append(image)
        weights.append(weight)
        bmasks.append(msk)
        bkgs.append(bkg)

    return {
        "info": info,
        "img": images,
        "wgt": weights,
        "msk": bmasks,
        "bkg": bkgs,
    }
