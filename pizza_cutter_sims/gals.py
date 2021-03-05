import logging
import numpy as np
import galsim

from .constants import MAGZP_REF

LOGGER = logging.getLogger(__name__)


def gen_gals(*, rng, layout_config, gal_config, pos_bounds):
    """Generate galaxies, positions and noise.

    Parameters
    ----------
    rng : np.random.RandomState
        An RNG to use for making galaxies.
    layout_config : dict
        A dictionary with info for the layout of the objects.
    gal_config : dict
        A dictionary with info for generating galaxies.
    pos_bounds : 2-tuple of floats
        The range in which to generate u and v.

    Returns
    -------
    gals : list of galsim.GSObject
        The set of galaxies to render in the image.
    upos : array of floats
        The u position in world coordinates.
    vpos : array of floats
        The v position in world coordinates.
    noise : float
        The pixel noise to apply to the image.
    """
    # possibly get the number density
    if gal_config["type"] in ["exp-bright", "exp-dim"]:
        n_gals = None
    else:
        raise ValueError("galaxy type '%s' not supported!" % gal_config["type"])

    # determine galaxy layout
    if layout_config["type"] == "grid":
        LOGGER.debug("using 'grid' layout")

        width = pos_bounds[1] - pos_bounds[0]
        delta = width / layout_config["ngal_per_side"]
        vpos, upos = np.mgrid[
            0:layout_config["ngal_per_side"],
            0:layout_config["ngal_per_side"]
        ]
        upos = upos.ravel() * delta + delta / 2 - width / 2
        vpos = vpos.ravel() * delta + delta / 2 - width / 2

        # dither
        dither_scale = layout_config["dither_scale"]
        upos += rng.uniform(low=-dither_scale/2, high=dither_scale/2)
        vpos += rng.uniform(low=-dither_scale/2, high=dither_scale/2)

        n_gals = layout_config["ngal_per_side"]**2

    elif layout_config["type"] == "random":
        LOGGER.debug("using 'random' layout")

        if n_gals is None:
            area = ((pos_bounds[1] - pos_bounds[0]) / 60.0)**2
            n_gals_mn = area * layout_config["ngal_per_arcmin2"]
            n_gals = rng.poisson(n_gals_mn)

        upos = rng.uniform(low=pos_bounds[0], high=pos_bounds[1], size=n_gals)
        vpos = rng.uniform(low=pos_bounds[0], high=pos_bounds[1], size=n_gals)
    else:
        raise ValueError(
            "galaxy layout type '%s' not supported!" % layout_config["type"]
        )

    # now get the objects
    if gal_config["type"] in ["exp-bright", "exp-dim"]:
        noise = gal_config["noise"]

        if gal_config["type"] == "exp-bright":
            mag = 18
        elif gal_config["type"] == "exp-dim":
            mag = 23.5
        flux = 10**(0.4 * (MAGZP_REF - mag))

        LOGGER.debug("using '%s' gal type w/ mag %s", gal_config["type"], mag)

        gals = []
        for u, v in zip(upos, vpos):
            gals.append(
                galsim.Sersic(
                    half_light_radius=0.5,
                    n=1,
                ).withFlux(flux)
            )
    else:
        raise ValueError("galaxy type '%s' not supported!" % gal_config["type"])

    return gals, upos, vpos, noise
