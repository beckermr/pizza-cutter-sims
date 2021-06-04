import logging
import numpy as np
import galsim
from hexalattice.hexalattice import create_hex_grid
from .glass import make_glass_layout
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
    if gal_config["type"] in ["exp-bright", "exp-dim", "exp-super-bright"]:
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

    elif layout_config["type"] == "hex":
        width = pos_bounds[1] - pos_bounds[0]
        delta = width / layout_config["ngal_per_side"]
        nx = int(layout_config["ngal_per_side"] * np.sqrt(2))
        # the factor of 0.866 makes sure the grid is square-ish
        ny = int(layout_config["ngal_per_side"] * np.sqrt(2) / 0.8660254)

        # here the spacing between grid centers is 1
        hg, _ = create_hex_grid(nx=nx, ny=ny, rotate_deg=rng.uniform() * 360)

        # convert the spacing to right number of pixels
        # we also recenter the grid since it comes out centered at 0,0
        hg *= delta
        upos = hg[:, 0].ravel()
        vpos = hg[:, 1].ravel()

        # dither
        dither_scale = layout_config["dither_scale"]
        upos += rng.uniform(low=-dither_scale/2, high=dither_scale/2)
        vpos += rng.uniform(low=-dither_scale/2, high=dither_scale/2)

        msk = (
            (upos >= pos_bounds[0])
            & (upos <= pos_bounds[1])
            & (vpos >= pos_bounds[0])
            & (vpos <= pos_bounds[1])
        )
        upos = upos[msk]
        vpos = vpos[msk]

        n_gals = upos.shape[0]

    elif layout_config["type"] == "glass":
        n_gals = layout_config["ngal_per_side"]**2

        upos, vpos = make_glass_layout(
            n_gals,
            pos_bounds,
            rng,
        )

        # this can return fewer than we asked for
        n_gals = upos.shape[0]
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
    if gal_config["type"] in ["exp-bright", "exp-dim", "exp-super-bright"]:
        noise = 10.0  # gal_config["noise"] <- this is ignored for these sims
        if "noise" in gal_config:
            LOGGER.critical(
                "The key 'noise' is in the gal config but it is being "
                "ignored and set to 10!"
            )

        if gal_config["type"] == "exp-super-bright":
            mag = 14.0  # snr ~ 2e4
        elif gal_config["type"] == "exp-bright":
            mag = 18.0  # snr ~ 500
        elif gal_config["type"] == "exp-dim":
            mag = 22.0  # snr ~ 10-12
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
