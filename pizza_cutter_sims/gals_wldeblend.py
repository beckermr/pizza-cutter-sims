import logging
import os
import copy
import functools
import collections
import numpy as np
import scipy.optimize

import galsim
import fitsio

LOGGER = logging.getLogger(__name__)
WLDeblendData = collections.namedtuple(
    'WLDeblendData',
    [
        'cat', 'survey_name', 'bands', 'surveys',
        'builders', 'total_sky', 'noise', 'ngal_per_arcmin2',
        'psf_fwhm', 'pixel_scale', 'flux_zeropoints',
        'flux_factor', 'round',
    ],
)

WLDeblendGal = collections.namedtuple(
    'WLDeblendGal',
    [
        'band_galaxies', 'color'
    ],
)


@functools.lru_cache(maxsize=8)
def _cached_catalog_read():
    fname = os.path.join(
        os.environ.get('CATSIM_DIR', '.'),
        'OneDegSq.fits',
    )
    return fitsio.read(fname)


@functools.lru_cache(maxsize=8)
def init_wldeblend(*, survey_bands):
    """Initialize weak lensing deblending survey data.

    Parameters
    ----------
    survey_bands : str
        The name of the survey followed by the bands like 'des-riz', 'lsst-iz', etc.

    Returns
    -------
    data : WLDeblendData
        Namedtuple with data for making galaxies via the weak lesning
        deblending package.
    """
    survey_name, bands = survey_bands.split("-")[0:2]
    bands = [b for b in bands]
    flux_factor = 1.0
    round_gal = False
    if len(survey_bands.split("-")) > 2:
        rest = survey_bands.split("-")[2:]
        for v in rest:
            if v.startswith("flux"):
                flux_factor = float(v[4:])
            elif v.startswith("round"):
                round_gal = True
    LOGGER.info('simulating survey: %s', survey_name)
    LOGGER.info('simulating bands: %s', bands)
    LOGGER.info('simulating flux factor: %f', flux_factor)
    LOGGER.info('simulating round: %s', str(round_gal))

    if survey_name not in ["des", "lsst"]:
        raise RuntimeError(
            "Survey for wldeblend must be one of 'des' or 'lsst'"
            " - got %s!" % survey_name
        )

    if survey_name == "lsst":
        scale = 0.2
    elif survey_name == "des":
        scale = 0.263

    # guard the import here
    import descwl

    # set the exposure times
    if survey_name == 'des':
        exptime = 90 * 10
    else:
        exptime = None

    wldeblend_cat = _cached_catalog_read()

    surveys = []
    builders = []
    total_sky = 0.0
    flux_zeropoints = []
    for iband, band in enumerate(bands):
        # make the survey and code to build galaxies from it
        pars = descwl.survey.Survey.get_defaults(
            survey_name=survey_name.upper(),
            filter_band=band)

        pars['survey_name'] = survey_name
        pars['filter_band'] = band
        pars['pixel_scale'] = scale

        # note in the way we call the descwl package, the image width
        # and height is not actually used
        pars['image_width'] = 100
        pars['image_height'] = 100

        # reset the exposure times if we want
        if exptime is not None:
            pars['exposure_time'] = exptime

        # some versions take in the PSF and will complain if it is not
        # given
        try:
            _svy = descwl.survey.Survey(**pars)
        except Exception:
            pars['psf_model'] = None
            _svy = descwl.survey.Survey(**pars)

        surveys.append(_svy)
        builders.append(descwl.model.GalaxyBuilder(
            survey=surveys[iband],
            no_disk=False,
            no_bulge=False,
            no_agn=False,
            verbose_model=False))

        total_sky += surveys[iband].mean_sky_level
        zp = scipy.optimize.brentq(
            lambda x: 1.0 - surveys[iband].get_flux(x),
            -100,
            100,
        )
        flux_zeropoints.append(zp)

    noise = np.sqrt(total_sky)

    if survey_name == "lsst":
        psf_fwhm = 0.7
    elif survey_name == "des":
        psf_fwhm = 0.9

    # when we sample from the catalog, we need to pull the right number
    # of objects. Since the default catalog is one square degree
    # and we fill a fraction of the image, we need to set the
    # base source density `ngal`. This is in units of number per
    # square arcminute.
    ngal_per_arcmin2 = wldeblend_cat.size / (60 * 60)

    LOGGER.info('catalog density: %f per sqr arcmin', ngal_per_arcmin2)

    return WLDeblendData(
        wldeblend_cat, survey_name, bands, surveys,
        builders, total_sky, noise, ngal_per_arcmin2,
        psf_fwhm, scale, flux_zeropoints, flux_factor, round_gal,
    )


def get_gal_wldeblend(*, rng, data):
    """Draw a galaxy from the weak lensing deblending package.

    Parameters
    ----------
    rng : np.random.RandomState
        An RNG to use for making galaxies.
    data : WLDeblendData
        Namedtuple with data for making galaxies via the weak lesning
        deblending package.

    Returns
    -------
    gal : galsim Object
        The galaxy as a galsim object.
    """
    rind = rng.choice(data.cat.size)
    angle = rng.uniform() * 360
    pa_angle = rng.uniform() * 360

    data.cat['pa_disk'][rind] = pa_angle
    data.cat['pa_bulge'][rind] = pa_angle

    entry = {
        k:  copy.deepcopy(data.cat[k][rind])
        for k in data.cat.dtype.names
    }

    if data.round:
        hlr_d = np.sqrt(entry['a_d'] * entry['b_d'])
        entry["a_d"] = hlr_d
        entry["b_d"] = hlr_d
        hlr_b = np.sqrt(entry['a_b'] * entry['b_b'])
        entry["a_b"] = hlr_b
        entry["b_b"] = hlr_b

    band_objects = [
        data.builders[band].from_catalog(
            entry,
            0,
            0,
            data.surveys[band].filter_band
        ).model.rotate(
            angle * galsim.degrees
        ) * data.flux_factor
        for band in range(len(data.builders))
    ]
    if len(band_objects) > 1:
        color = (
            data.cat[f"{data.bands[0]}_ab"][rind]
            - data.cat[f"{data.bands[1]}_ab"][rind]
        )
    else:
        color = 0

    return WLDeblendGal(
        band_objects,
        color,
    )


def get_psf_config_wldeblend(*, data):
    """Get a config dict for a the PSF model for the weak lensing deblending
    objects.

    Parameters
    ----------
    data : WLDeblendData
        Namedtuple with data for making galaxies via the weak lesning
        deblending package.

    Returns
    -------
    gs_config : dict
        A dictionary with the PSF info.
    """
    gs_config = {}
    gs_config["type"] = "Moffat"
    gs_config["beta"] = 2.5
    gs_config["fwhm"] = data.psf_fwhm
    return gs_config
