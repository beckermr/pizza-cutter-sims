import numpy as np

from pizza_cutter.des_pizza_cutter import load_objects_into_info
from pizza_cutter.des_pizza_cutter._coadd_slices import (
    _build_slice_inputs, _coadd_slice_inputs)


def run_des_pizza_cutter_coadding_on_sim(
    *, rng, tmpdir, info, img, wgt, msk, bkg, single_epoch_config
):
    """Run pizza cutter coadding on a sim.

    Parameters
    ----------
    rng : np.random.RandomState
        An RNG to use for running coadding.
    tmpdir : str
        A temporary directory to use during coadding.
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
    single_epoch_config : dict
        A dictionary of config information for how the coadd treats the single
        epoch images.

    Returns
    -------
    coadd_data : dict
        A dictionary with keys:

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
    """
    # this creates the WCS objects
    load_objects_into_info(info=info, verbose=False)

    for ii in info["src_info"]:
        ii["noise_seed"] = rng.randint(low=1, high=2**29)

    # set the object config info
    object_config = {
        "ra": 0.0,
        "dec": 0.0,
        "position_offset": info["position_offset"],
        "psf_box_size": 51,
        "box_size": info["image_shape"][0],
        "orig_start_col": 0,
        "orig_start_row": 0,
        "coadding_weight": "noise",
    }
    single_epoch_config["psf_type"] = "galsim"
    single_epoch_config["wcs_type"] = "affine"

    # we center the PSF at the nearest pixel center near the patch center
    wcs = info["affine_wcs"]
    col, row = wcs.sky2image(object_config['ra'], object_config['dec'])
    # this col, row includes the position offset
    # we don't need to remove it when putting them back into the WCS
    # but we will remove it later since we work in zero-indexed coords
    col = int(col + 0.5)
    row = int(row + 0.5)
    # ra, dec of the pixel center
    ra_psf, dec_psf = wcs.image2sky(col, row)

    # now we find the lower left location of the PSF image
    half = (object_config['psf_box_size'] - 1) / 2
    assert int(half) == half, "PSF images must have odd dimensions!"
    # here we remove the position offset
    col -= object_config['position_offset']
    row -= object_config['position_offset']
    psf_orig_start_col = col - half
    psf_orig_start_row = row - half

    bsres = _build_slice_inputs(
        ra=object_config['ra'],
        dec=object_config['dec'],
        ra_psf=ra_psf,
        dec_psf=dec_psf,
        box_size=object_config['box_size'],
        coadd_info=info,
        start_row=object_config['orig_start_row'],
        start_col=object_config['orig_start_col'],
        se_src_info=info['src_info'],
        reject_outliers=single_epoch_config['reject_outliers'],
        symmetrize_masking=single_epoch_config['symmetrize_masking'],
        coadding_weight=object_config['coadding_weight'],
        noise_interp_flags=sum(single_epoch_config['noise_interp_flags']),
        spline_interp_flags=sum(
            single_epoch_config['spline_interp_flags']),
        bad_image_flags=sum(single_epoch_config['bad_image_flags']),
        max_masked_fraction=single_epoch_config['max_masked_fraction'],
        max_unmasked_trail_fraction=single_epoch_config[
            'max_unmasked_trail_fraction'],
        mask_tape_bumps=single_epoch_config['mask_tape_bumps'],
        edge_buffer=single_epoch_config['edge_buffer'],
        wcs_type=single_epoch_config['wcs_type'],
        psf_type=single_epoch_config['psf_type'],
        rng=rng,
        tmpdir=tmpdir,
    )

    se_image_slices, weights, slices_not_used, flags_not_used = bsres

    # did we get anything?
    if np.array(weights).size > 0:
        image, bmask, ormask, noise, psf, weight = _coadd_slice_inputs(
            wcs=wcs,
            wcs_position_offset=object_config['position_offset'],
            start_row=object_config['orig_start_row'],
            start_col=object_config['orig_start_col'],
            box_size=object_config['box_size'],
            psf_start_row=psf_orig_start_row,
            psf_start_col=psf_orig_start_col,
            psf_box_size=object_config['psf_box_size'],
            se_image_slices=se_image_slices,
            weights=weights)

        coadd_data = dict(
            image=image,
            bmask=bmask,
            ormask=ormask,
            noise=noise,
            psf=psf,
            weight=weight,
        )
        return coadd_data
    else:
        raise RuntimeError("No images with positive weight were found!")