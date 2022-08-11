import logging

import numpy as np

from .gsutils import build_gsobject
from .ps_psf import PowerSpectrumPSF
from . import gals_wldeblend

LOGGER = logging.getLogger(__name__)


class GalsimPSF(object):
    """Make a PSF based on a galsim object.

    Parameters
    ----------
    gs_object : galsim.GSObject
        The PSF object to return.
    color_dep_fun : function or None
        A function used to change the size of the PSF with color.
    """
    def __init__(self, gs_object, color_dep_fun=None):
        self.gs_object = gs_object
        self.color_dep_fun = color_dep_fun

    def getPSF(self, pos, color=None):
        """Get a PSF model at a given position.

        Parameters
        ----------
        pos : galsim.PositionD
            The position at which to compute the PSF. In zero-indexed
            pixel coordinates.
        color : float or None, optional
            Used to change the PSF size with color.

        Returns
        -------
        psf : galsim.GSObject
            A representation of the PSF as a galism object.
        """
        if color is not None and self.color_dep_fun is not None:
            pdf = self.color_dep_fun(color)
            return self.gs_object.dilate(pdf)
        else:
            return self.gs_object

    def __eq__(self, other):
        return self.gs_object == other.gs_object


def gen_psf(*, rng, psf_config, gal_config, se_image_shape, se_wcs):
    """Generate the PSF for the SE image.

    Parameters
    ----------
    rng : np.random.RandomState
        An RNG instance to use.
    psf_config : dict
        A dictionary of the PSF config information.
    gal_config : dict
        A dictionary of the galaxy config information.
    se_image_shape : int
        The shape of the SE image.
    se_wcs : galsim WCS object
        The WCS for the SE image.

    Returns
    -------
    gs_config : dict
        A config dict for the galsim object for the image's center.
    psf_klass : object
        An object to draw PSFs at different positions in the image.
    """
    if psf_config["type"].startswith("galsim."):
        g1 = 1.1
        g2 = 1.1
        while np.abs(g1) > 1 or np.abs(g2) > 1 or np.abs(g1*g1 + g2*g2) > 1:
            g1 = (
                psf_config["shear"][0] + rng.normal() * psf_config["shear_std"]
            )
            g2 = (
                psf_config["shear"][1] + rng.normal() * psf_config["shear_std"]
            )
        width = psf_config["fwhm"] * (
            1.0 + psf_config["fwhm_frac_std"] * rng.normal()
        )

        kwargs = {
            k: v
            for k, v in psf_config.items()
            if k not in ("shear", "shear_std", "fwhm", "fwhm_frac_std", "type")
        }

        gs_config = {}
        gs_config.update(kwargs)
        gs_config.pop("color_range", None)
        gs_config.pop("dilation_range", None)
        gs_config["type"] = psf_config["type"].replace("galsim.", "")
        gs_config["fwhm"] = width
        gs_config["shear"] = {
            "type": "G1G2",
            "g1": g1,
            "g2": g2,
        }

        def _color_dep_fun(color):
            if color < psf_config["color_range"][0]:
                color = psf_config["color_range"][0]
            if color > psf_config["color_range"][1]:
                color = psf_config["color_range"][1]

            w = (
                (color - psf_config["color_range"][0])
                / (psf_config["color_range"][1] - psf_config["color_range"][0])
            )
            return (
                psf_config["dilation_range"][0] * (1-w)
                + psf_config["dilation_range"][1] * w
            )

        res = gs_config, GalsimPSF(
            build_gsobject(config=gs_config, kind='psf'),
            color_dep_fun=_color_dep_fun,
        )
        LOGGER.debug("psf config: %s", res[0])
        LOGGER.debug("galsim psf: %s", res[1].gs_object)
        return res
    elif psf_config["type"] == "ps":
        psf_obj = PowerSpectrumPSF(
            rng=rng,
            im_width=se_image_shape,
            buff=20,
            trunc=1,
            scale=np.sqrt(se_wcs.pixelArea(image_pos=se_wcs.origin)),
            variation_factor=psf_config["variation_factor"],
            fwhm=psf_config["fwhm"],
        )
        gs_config = psf_obj.getPSFConfig(se_wcs.origin)
        LOGGER.debug("ps psf config: %s", gs_config)
        LOGGER.debug("ps galsim psf: %s", psf_obj.getPSF(se_wcs.origin))
        return gs_config, psf_obj
    elif psf_config["type"] == "wldeblend":
        data = gals_wldeblend.init_wldeblend(survey_bands=gal_config["type"])
        gs_config = gals_wldeblend.get_psf_config_wldeblend(data=data)

        g1 = 1.1
        g2 = 1.1
        while np.abs(g1) > 1 or np.abs(g2) > 1 or np.abs(g1*g1 + g2*g2) > 1:
            g1 = (
                psf_config["shear"][0] + rng.normal() * psf_config["shear_std"]
            )
            g2 = (
                psf_config["shear"][1] + rng.normal() * psf_config["shear_std"]
            )
        width = gs_config["fwhm"] * (
            1.0 + psf_config["fwhm_frac_std"] * rng.normal()
        )

        gs_config["fwhm"] = width
        gs_config["shear"] = {
            "type": "G1G2",
            "g1": g1,
            "g2": g2,
        }

        def _color_dep_fun(color):
            if color < psf_config["color_range"][0]:
                color = psf_config["color_range"][0]
            if color > psf_config["color_range"][1]:
                color = psf_config["color_range"][1]

            w = (
                (color - psf_config["color_range"][0])
                / (psf_config["color_range"][1] - psf_config["color_range"][0])
            )
            return (
                psf_config["dilation_range"][0] * (1-w)
                + psf_config["dilation_range"][1] * w
            )

        res = gs_config, GalsimPSF(
            build_gsobject(config=gs_config, kind='psf'),
            color_dep_fun=_color_dep_fun,
        )
        LOGGER.debug("wldeblend psf config: %s", res[0])
        LOGGER.debug("wldeblend galsim psf: %s", res[1].gs_object)
        return res
    elif psf_config["type"] == "wldeblend-ps":
        data = gals_wldeblend.init_wldeblend(survey_bands=gal_config["type"])
        gs_config = gals_wldeblend.get_psf_config_wldeblend(data=data)

        psf_obj = PowerSpectrumPSF(
            rng=rng,
            im_width=se_image_shape,
            buff=20,
            trunc=1,
            scale=np.sqrt(se_wcs.pixelArea(image_pos=se_wcs.origin)),
            variation_factor=psf_config["variation_factor"],
            fwhm=gs_config["fwhm"],
        )
        gs_config = psf_obj.getPSFConfig(se_wcs.origin)
        LOGGER.debug("wldeblend-ps psf config: %s", gs_config)
        LOGGER.debug("wldeblend-ps galsim psf: %s", psf_obj.getPSF(se_wcs.origin))
        return gs_config, psf_obj
    else:
        raise ValueError("PSF type '%s' not supported!" % psf_config["type"])
