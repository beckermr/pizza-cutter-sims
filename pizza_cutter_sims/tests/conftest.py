import numpy as np
import pytest
import yaml


def recursive_equal(sdata1, sdata2):
    eq = True
    if isinstance(sdata1, np.ndarray):
        if hasattr(sdata1.dtype, "names") and sdata1.dtype.names is not None:
            for name in sdata1.dtype.names:
                eq = eq and recursive_equal(sdata1[name], sdata2[name])
        else:
            try:
                np.testing.assert_array_equal(sdata1, sdata2)
                _eq = True
            except AssertionError:
                _eq = False
            eq = eq and _eq
    elif isinstance(sdata1, dict):
        for k in sdata1:
            eq = eq and recursive_equal(sdata1[k], sdata2[k])
    elif isinstance(sdata1, list) or isinstance(sdata1, tuple):
        for item1, item2 in zip(sdata1, sdata2):
            eq = eq and recursive_equal(item1, item2)
    else:
        eq = eq and (sdata1 == sdata2)
    return eq


@pytest.fixture()
def sim_config():
    cfg = """\
shear:
  scene: True

  # these keys are used to build paired sims w/ noise cancellation
  g: 0.02
  swap12: False

  # set these to run a sim directly
  g1: 0.02
  g2: 0.00

msk:
  cosmic_rays: False
    # mean_cosmic_rays: 1
    # min_length: 10
    # max_length: 30

  bad_columns: False
    # mean_bad_cols: 1
    # widths: [1, 2, 5, 10]
    # p: [0.8, 0.1, 0.075, 0.025]
    # min_length_frac: [1, 1, 0.25, 0.25]
    # max_length_frac: [1, 1, 0.75, 0.75]
    # gap_prob: [0.30, 0.30, 0, 0]
    # min_gap_frac: [0.1, 0.1, 0, 0]
    # max_gap_frac: [0.3, 0.3, 0, 0]

  streaks: False

coadd:
  central_size: 225
  buffer_size: 25
  scale: 0.263

se:
  n_images: 1
  wcs_config:
    position_angle_range: [0, 0]
    dither_scale: 0  # set up to a pizel in arcsec
    scale: 0.263
    scale_frac_std: 0
    shear_std: 0
  residual_bkg: 0.0
  residual_bkg_std: 0.0

psf:
  type: galsim.Gaussian
  fwhm: 0.9
  fwhm_frac_std: 0.1
  shear_std: 0
  shear: [0, 0]
  color_range: [0, 3]
  dilation_range: [0.8, 1.1]

layout:
  type: grid
  ngal_per_side: 7
  ngal_per_arcmin2: 60
  dither_scale: 0.263

gal:
  type: exp-bright
  multiband: False
  color_range: [1, 1]
  color_mean: 1
  color_std: 1
  color_type: uniform

star:
  dens_factor: 1
  rad_dist: uniform
  interp:
    # these control how the interpolation is applied for star wholes
    # if fill_isolated_with_noise is True, then any missing pixel with no non-missing
    # pixels within iso_buff will be filled with noise and then used to interpolate
    # the rest of the pixels.
    skip: False
    iso_buff: 1
    fill_isolated_with_noise: True
  apodize:
    ap_rad: 1
    skip: True
  mask_expand_rad: 0

pizza_cutter:
  skip: False
  single_epoch_config:
    se_wcs_interp_delta: 10
    coadd_wcs_interp_delta: 25
    frac_buffer: 1

    reject_outliers: False
    symmetrize_masking: True
    copy_masked_edges: True
    max_masked_fraction: 0.1
    max_unmasked_trail_fraction: 0.02
    edge_buffer: 8
    mask_tape_bumps: False

    # set the interp flags to 0 or 3 to interp stuff in the mask
    spline_interp_flags:
      - 0
    noise_interp_flags:
      - 0

    # always zero
    bad_image_flags:
      - 0

metadetect:
  color_dep_psf:
    skip: True

  metacal:
    psf: fitgauss
    types: [noshear, 1p, 1m, 2p, 2m]
    use_noise_image: True

  psf:
    lm_pars:
      maxfev: 2000
      ftol: 1.0e-05
      xtol: 1.0e-05
    model: gauss

    # we try many times because if this fails we get no psf info
    # for the entire patch
    ntry: 10

  sx:
    # Minimum contrast parameter for deblending
    deblend_cont: 1.0e-05

    # in sky sigma
    detect_thresh: 0.8

    # minimum number of pixels above threshold
    minarea: 4

    filter_type: conv
    # 7x7 convolution mask of a gaussian PSF with FWHM = 3.0 pixels.
    filter_kernel:
      - [0.004963, 0.021388, 0.051328, 0.068707, 0.051328, 0.021388, 0.004963]
      - [0.021388, 0.092163, 0.221178, 0.296069, 0.221178, 0.092163, 0.021388]
      - [0.051328, 0.221178, 0.530797, 0.710525, 0.530797, 0.221178, 0.051328]
      - [0.068707, 0.296069, 0.710525, 0.951108, 0.710525, 0.296069, 0.068707]
      - [0.051328, 0.221178, 0.530797, 0.710525, 0.530797, 0.221178, 0.051328]
      - [0.021388, 0.092163, 0.221178, 0.296069, 0.221178, 0.092163, 0.021388]
      - [0.004963, 0.021388, 0.051328, 0.068707, 0.051328, 0.021388, 0.004963]

  weight:
    fwhm: 1.2  # arcsec

  meds:
    box_padding: 2
    box_type: iso_radius
    max_box_size: 64
    min_box_size: 32
    rad_fac: 2
    rad_min: 4

  # check for an edge hit
  bmask_flags: 536870912  # 2**29

  # flags for mask fractions computed by mdet
  star_flags: 0
  tapebump_flags: 0
  spline_interp_flags: 0
  noise_interp_flags: 0
  imperfect_flags: 0
"""
    return yaml.safe_load(cfg)
