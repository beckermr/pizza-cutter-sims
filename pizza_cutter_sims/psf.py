import logging

import numpy as np

from .gsutils import build_gsobject
from .ps_psf import PowerSpectrumPSF

LOGGER = logging.getLogger(__name__)


class GalsimPSF(object):
    """Make a PSF based on a galsim object.

    Parameters
    ----------
    gs_object : galsim.GSObject
        The PSF object to return.
    """
    def __init__(self, gs_object):
        self.gs_object = gs_object

    def getPSF(self, pos):
        """Get a PSF model at a given position.

        Parameters
        ----------
        pos : galsim.PositionD
            The position at which to compute the PSF. In zero-indexed
            pixel coordinates.

        Returns
        -------
        psf : galsim.GSObject
            A representation of the PSF as a galism object.
        """
        return self.gs_object

    def __eq__(self, other):
        return self.gs_object == other.gs_object


def gen_psf(*, rng, psf_config, se_image_shape, se_wcs):
    """Generate the PSF for the SE image.

    Parameters
    ----------
    rng : np.random.RandomState
        An RNG instance to use.
    psf_config : dict
        A dictionary of the PSF config information.
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
        gs_config["type"] = psf_config["type"].replace("galsim.", "")
        gs_config["fwhm"] = width
        gs_config["shear"] = {
            "type": "G1G2",
            "g1": g1,
            "g2": g2,
        }

        res = gs_config, GalsimPSF(
            build_gsobject(config=gs_config, kind='psf')
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
        LOGGER.debug("psf config: %s", gs_config)
        LOGGER.debug("galsim psf: %s", psf_obj.getPSF(se_wcs.origin))
        return gs_config, psf_obj
    else:
        raise ValueError("PSF type '%s' not supported!" % psf_config["type"])
