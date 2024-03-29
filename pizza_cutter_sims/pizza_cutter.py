import numpy as np

from pizza_cutter.des_pizza_cutter import load_objects_into_info
from pizza_cutter.des_pizza_cutter._coadd_slices import (
    _build_slice_inputs, _coadd_slice_inputs)


def run_des_pizza_cutter_coadding_on_sim(
    *, rng, tmpdir, info, img, wgt, msk, bkg, single_epoch_config,
    n_extra_noise_images,
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
    n_extra_noise_images : int
        The number of extra noise images to make.

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
            The noise image for the coadd.
        psf : np.ndarray
            The coadd PSF image.
        weight : np.ndarray
            The weight map for the coadd.
        mfrac : np.ndarray
            The fraction of SE images in each pixel that is masked.
        extra_noises : list of np.ndarray
            A list of noise images for the coadd.
    """
    # this creates the WCS objects
    load_objects_into_info(info=info, verbose=False)

    for ii in info["src_info"]:
        ii["noise_seeds"] = rng.randint(low=1, high=2**29, size=5)

    # set the object config info
    object_config = {
        "ra": 0.0,
        "dec": 0.0,
        "position_offset": info["position_offset"],
        "psf_box_size": 101,
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
        wcs=info["affine_wcs"],
        wcs_position_offset=info["position_offset"],
        wcs_interp_delta=single_epoch_config["se_wcs_interp_delta"],
        box_size=object_config['box_size'],
        frac_buffer=single_epoch_config['frac_buffer'],
        start_row=object_config['orig_start_row'],
        start_col=object_config['orig_start_col'],
        se_src_info=info['src_info'],
        reject_outliers=single_epoch_config['reject_outliers'],
        symmetrize_masking=single_epoch_config['symmetrize_masking'],
        copy_masked_edges=single_epoch_config['copy_masked_edges'],
        coadding_weight=object_config['coadding_weight'],
        noise_interp_flags=sum(single_epoch_config['noise_interp_flags']),
        spline_interp_flags=sum(
            single_epoch_config['spline_interp_flags']),
        bad_image_flags=sum(single_epoch_config['bad_image_flags']),
        max_masked_fraction=single_epoch_config['max_masked_fraction'],
        mask_tape_bumps=single_epoch_config['mask_tape_bumps'],
        edge_buffer=single_epoch_config['edge_buffer'],
        wcs_type=single_epoch_config['wcs_type'],
        wcs_color=0.0,  # we never deal with this kind of thing
        psf_type=single_epoch_config['psf_type'],
        psf_kwargs={},
        rng=rng,
        tmpdir=tmpdir,
        n_extra_noise_images=n_extra_noise_images,
        mask_piff_failure_config=None,
    )

    se_image_slices, weights, slices_not_used, flags_not_used = bsres

    # did we get anything?
    if np.array(weights).size > 0:
        res = _coadd_slice_inputs(
            wcs=wcs,
            wcs_position_offset=object_config['position_offset'],
            wcs_image_shape=info["image_shape"],
            start_row=object_config['orig_start_row'],
            start_col=object_config['orig_start_col'],
            box_size=object_config['box_size'],
            psf_start_row=psf_orig_start_row,
            psf_start_col=psf_orig_start_col,
            psf_box_size=object_config['psf_box_size'],
            se_image_slices=se_image_slices,
            weights=weights,
            se_wcs_interp_delta=single_epoch_config["se_wcs_interp_delta"],
            coadd_wcs_interp_delta=single_epoch_config["coadd_wcs_interp_delta"],
            n_extra_noise_images=n_extra_noise_images,
        )
        image, bmask, ormask, noises, psf, weight, mfrac, rsd = (
            res[0],
            res[1],
            res[2],
            res[3],
            res[4],
            res[5],
            res[6],
            res[-1]
        )
        coadd_data = dict(
            image=image,
            bmask=bmask,
            ormask=ormask,
            noise=noises[0],
            psf=psf,
            weight=weight,
            rsd=rsd,
            mfrac=mfrac,
        )
        if n_extra_noise_images > 0:
            coadd_data["extra_noises"] = noises[1:]
        return coadd_data
    else:
        raise RuntimeError("No images with positive weight were found!")


def _make_coadd_slice_inputs(
    *, wcs, wcs_position_offset, wcs_image_shape, start_row, start_col,
    box_size, psf_start_row, psf_start_col, psf_box_size,
    se_image_slices, weights, se_wcs_interp_delta, coadd_wcs_interp_delta,
    n_extra_noise_images,
):
    # normalize just in case
    weights = np.atleast_1d(weights) / np.sum(weights)

    # make sure input data is consistent
    assert len(se_image_slices) == len(weights), (
        "The input set of weights and images are different sizes.")

    resampled_datas = []

    for se_slice, weight in zip(se_image_slices, weights):
        resampled_data = se_slice.resample(
            wcs=wcs,
            wcs_position_offset=wcs_position_offset,
            wcs_interp_shape=wcs_image_shape,
            x_start=start_col,
            y_start=start_row,
            box_size=box_size,
            psf_x_start=psf_start_col,
            psf_y_start=psf_start_row,
            psf_box_size=psf_box_size,
            se_wcs_interp_delta=se_wcs_interp_delta,
            coadd_wcs_interp_delta=coadd_wcs_interp_delta,
        )
        resampled_data["single_epoch_psf"] = se_slice.psf.copy()
        resampled_datas.append(resampled_data)

    return resampled_datas


def make_remapped_se_images_des_pizza_cutter_coadding_on_sim(
    *, rng, tmpdir, info, img, wgt, msk, bkg, single_epoch_config,
    n_extra_noise_images,
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
    n_extra_noise_images : int
        The number of extra noise images to make.

    Returns
    -------
    resampled_data : list of dicts
        A list of dicts of the resampled data for each inpout epoch.
    weights : array
        The normalized weight to apply to each image.
    """
    # this creates the WCS objects
    load_objects_into_info(info=info, verbose=False)

    for ii in info["src_info"]:
        ii["noise_seeds"] = rng.randint(low=1, high=2**29, size=5)

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
        wcs=info["affine_wcs"],
        wcs_position_offset=info["position_offset"],
        wcs_interp_delta=single_epoch_config["se_wcs_interp_delta"],
        box_size=object_config['box_size'],
        frac_buffer=single_epoch_config['frac_buffer'],
        start_row=object_config['orig_start_row'],
        start_col=object_config['orig_start_col'],
        se_src_info=info['src_info'],
        reject_outliers=single_epoch_config['reject_outliers'],
        symmetrize_masking=single_epoch_config['symmetrize_masking'],
        copy_masked_edges=single_epoch_config['copy_masked_edges'],
        coadding_weight=object_config['coadding_weight'],
        noise_interp_flags=sum(single_epoch_config['noise_interp_flags']),
        spline_interp_flags=sum(
            single_epoch_config['spline_interp_flags']),
        bad_image_flags=sum(single_epoch_config['bad_image_flags']),
        max_masked_fraction=single_epoch_config['max_masked_fraction'],
        mask_tape_bumps=single_epoch_config['mask_tape_bumps'],
        edge_buffer=single_epoch_config['edge_buffer'],
        wcs_type=single_epoch_config['wcs_type'],
        wcs_color=0.0,  # we never deal with this kind of thing
        psf_type=single_epoch_config['psf_type'],
        psf_kwargs={},
        rng=rng,
        tmpdir=tmpdir,
        n_extra_noise_images=n_extra_noise_images,
        mask_piff_failure_config=None,
    )

    se_image_slices, weights, slices_not_used, flags_not_used = bsres

    # did we get anything?
    if np.array(weights).size > 0:
        res = _make_coadd_slice_inputs(
            wcs=wcs,
            wcs_position_offset=object_config['position_offset'],
            wcs_image_shape=info["image_shape"],
            start_row=object_config['orig_start_row'],
            start_col=object_config['orig_start_col'],
            box_size=object_config['box_size'],
            psf_start_row=psf_orig_start_row,
            psf_start_col=psf_orig_start_col,
            psf_box_size=object_config['psf_box_size'],
            se_image_slices=se_image_slices,
            weights=weights,
            se_wcs_interp_delta=single_epoch_config["se_wcs_interp_delta"],
            coadd_wcs_interp_delta=single_epoch_config["coadd_wcs_interp_delta"],
            n_extra_noise_images=n_extra_noise_images,
        )
        return res, weights / np.sum(weights)
    else:
        raise RuntimeError("No images with positive weight were found!")
