import logging
import numpy as np
import galsim
import copy
from hexalattice.hexalattice import create_hex_grid
from .glass import make_glass_layout
from .constants import MAGZP_REF
from . import gals_wldeblend

LOGGER = logging.getLogger(__name__)


DES_GMI_BE = np.array([
    -2.8886816, -2.79877759, -2.70887357, -2.61896955, -2.52906553,
    -2.43916152, -2.3492575, -2.25935348, -2.16944947, -2.07954545,
    -1.98964143, -1.89973741, -1.8098334, -1.71992938, -1.63002536,
    -1.54012135, -1.45021733, -1.36031331, -1.27040929, -1.18050528,
    -1.09060126, -1.00069724, -0.91079323, -0.82088921, -0.73098519,
    -0.64108117, -0.55117716, -0.46127314, -0.37136912, -0.28146511,
    -0.19156109, -0.10165707, -0.01175305,  0.07815096,  0.16805498,
    0.257959,  0.34786301,  0.43776703,  0.52767105, 0.61757506,
    0.70747908,  0.7973831, 0.88728712,  0.97719113,  1.06709515,
    1.15699917,  1.24690318,  1.3368072, 1.42671122,  1.51661524,
    1.60651925,  1.69642327,  1.78632729,  1.8762313, 1.96613532,
    2.05603934,  2.14594336,  2.23584737,  2.32575139,  2.41565541,
    2.50555942,  2.59546344,  2.68536746,  2.77527148,  2.86517549,
    2.95507951,  3.04498353,  3.13488754,  3.22479156,  3.31469558,
    3.4045996, 3.49450361,  3.58440763,  3.67431165,  3.76421566,
    3.85411968,  3.9440237, 4.03392771,  4.12383173,  4.21373575,
    4.30363977,  4.39354378,  4.4834478, 4.57335182,  4.66325583,
    4.75315985,  4.84306387,  4.93296789,  5.0228719, 5.11277592,
    5.20267994,  5.29258395,  5.38248797,  5.47239199,  5.56229601,
    5.65220002,  5.74210404,  5.83200806,  5.92191207,  6.01181609,
    6.10172011
])


DES_GMI_CDF = np.array([
    0.00000000e+00, 2.30887960e-06, 2.30887960e-06, 2.30887960e-06,
    4.61775921e-06, 9.23551842e-06, 1.03899582e-05, 1.38532776e-05,
    2.30887960e-05, 3.57876339e-05, 5.42586707e-05, 6.46486289e-05,
    8.65829852e-05, 1.22370619e-04, 1.66239332e-04, 2.36660159e-04,
    3.31324223e-04, 4.44459324e-04, 5.89918739e-04, 7.95409024e-04,
    1.02052479e-03, 1.30451698e-03, 1.68317323e-03, 2.18073679e-03,
    2.86416515e-03, 3.65380197e-03, 4.75051979e-03, 6.23166605e-03,
    8.22653803e-03, 1.08944484e-02, 1.46336789e-02, 1.96716542e-02,
    2.67806945e-02, 3.67319656e-02, 5.01130774e-02, 6.75335740e-02,
    8.94933279e-02, 1.17302628e-01, 1.50583973e-01, 1.89486286e-01,
    2.36229553e-01, 2.88100843e-01, 3.45242150e-01, 4.03848441e-01,
    4.64133287e-01, 5.25280500e-01, 5.82576502e-01, 6.35287069e-01,
    6.82508274e-01, 7.22752046e-01, 7.57407174e-01, 7.86948134e-01,
    8.12696760e-01, 8.34314800e-01, 8.54055720e-01, 8.71745201e-01,
    8.87671853e-01, 9.02349400e-01, 9.15968327e-01, 9.28925759e-01,
    9.40861512e-01, 9.51694775e-01, 9.60682089e-01, 9.68386820e-01,
    9.74789344e-01, 9.80054744e-01, 9.84427762e-01, 9.87958038e-01,
    9.90974590e-01, 9.93310021e-01, 9.95089013e-01, 9.96529754e-01,
    9.97524881e-01, 9.98170213e-01, 9.98704719e-01, 9.99124935e-01,
    9.99414699e-01, 9.99613263e-01, 9.99742560e-01, 9.99830297e-01,
    9.99891483e-01, 9.99936506e-01, 9.99965367e-01, 9.99971139e-01,
    9.99981529e-01, 9.99983838e-01, 9.99987301e-01, 9.99991919e-01,
    9.99996537e-01, 9.99996537e-01, 9.99996537e-01, 9.99996537e-01,
    9.99996537e-01, 9.99996537e-01, 9.99996537e-01, 9.99996537e-01,
    9.99996537e-01, 9.99996537e-01, 9.99996537e-01, 9.99998846e-01,
    1.00000000e+00
])


def _draw_des_gmi(n, rng):
    x = rng.uniform(size=n)
    return np.interp(x, DES_GMI_CDF, DES_GMI_BE)


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
    gals : list of list of galsim.GSObject
        The set of galaxies to render in the image. Each element of the inner list is
        the object in a given band.
    upos : array of floats
        The u position in world coordinates.
    vpos : array of floats
        The v position in world coordinates.
    noise : float
        The pixel noise to apply to the image.
    noise_scale : float
        The pixel scale associated with the noise level. This can be used to
        adjust the noise for varying pixel sizes. Can be None.
    colors : list of floats or list of Nones
        The color of each object.
    flux_zeropoints : list of floats
        The flux zero points for each bands.
    """
    # possibly get the number density
    if gal_config["type"] in ["exp-bright", "exp-dim", "exp-super-bright"]:
        n_gals = None
    elif (
        gal_config["type"].startswith("des-")
        or gal_config["type"].startswith("lsst-")
    ):
        data = gals_wldeblend.init_wldeblend(survey_bands=gal_config["type"])
        area = ((pos_bounds[1] - pos_bounds[0]) / 60.0)**2
        n_gals_mn = area * data.ngal_per_arcmin2
        n_gals = rng.poisson(n_gals_mn)
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

        n_gals = layout_config["ngal_per_side"]**2

        # dither
        dither_scale = layout_config["dither_scale"]
        upos += rng.uniform(low=-dither_scale/2, high=dither_scale/2, size=n_gals)
        vpos += rng.uniform(low=-dither_scale/2, high=dither_scale/2, size=n_gals)

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
        n = upos.shape[0]
        dither_scale = layout_config["dither_scale"]
        upos += rng.uniform(low=-dither_scale/2, high=dither_scale/2, size=n)
        vpos += rng.uniform(low=-dither_scale/2, high=dither_scale/2, size=n)

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
        noise_scale = None

        if gal_config["type"] == "exp-super-bright":
            mag = 14.0  # snr ~ 2e4
        elif gal_config["type"] == "exp-bright":
            mag = 18.0  # snr ~ 500
        elif gal_config["type"] == "exp-dim":
            mag = 22.0  # snr ~ 10-12
        flux = 10**(0.4 * (MAGZP_REF - mag))

        LOGGER.debug("using '%s' gal type w/ mag %s", gal_config["type"], mag)

        if gal_config["multiband"]:
            nbands = gal_config["multiband"]
        else:
            nbands = 1

        if gal_config["color_type"] == "des":
            LOGGER.info("using DES galaxy color distribution")
            colors = [c for c in _draw_des_gmi(len(upos), rng)]
        elif gal_config["color_type"] == "lognormal":
            LOGGER.info("using lognormal galaxy color distribution")
            colors = [
                np.clip(
                    np.exp(rng.normal(
                        loc=np.log(gal_config["color_mean"]),
                        scale=gal_config["color_std"],
                    )),
                    gal_config["color_range"][0],
                    gal_config["color_range"][1],
                )
                for _ in range(len(upos))
            ]
        elif gal_config["color_type"] == "uniform":
            LOGGER.info("using uniform galaxy color distribution")
            colors = [
                c for c in rng.uniform(
                    low=gal_config["color_range"][0],
                    high=gal_config["color_range"][1],
                    size=len(upos)
                )
            ]
        else:
            raise RuntimeError(
                "gal config color type '%s' not recognized!" % gal_config["color_type"]
            )

        gals = []
        for u, v, color in zip(upos, vpos, colors):
            flux_ratios = [1.0] * nbands
            if nbands > 1:
                flux_ratio = 10.0**(0.4 * color)
                flux_ratios[1] = flux_ratio
                flux_ratios = (
                    np.array(flux_ratios)
                    / np.sum(flux_ratios)
                    * len(flux_ratios)
                )

            gals.append([
                galsim.Sersic(
                    half_light_radius=0.5,
                    n=1,
                ).withFlux(flux * fr)
                for fr in flux_ratios
            ])

        flux_zeropoints = [MAGZP_REF] * nbands
    elif (
        gal_config["type"].startswith("des-")
        or gal_config["type"].startswith("lsst-")
    ):
        data = gals_wldeblend.init_wldeblend(survey_bands=gal_config["type"])
        gals = []
        colors = []
        for _ in range(n_gals):
            res = gals_wldeblend.get_gal_wldeblend(
                rng=rng,
                data=data,
            )
            gals.append(res.band_galaxies)
            colors.append(res.color)
        noise = copy.copy(data.noise)
        noise_scale = copy.copy(data.pixel_scale)
        flux_zeropoints = data.flux_zeropoints
    else:
        raise ValueError("galaxy type '%s' not supported!" % gal_config["type"])

    LOGGER.debug("simulated %d galaxies for galaxy type %s", n_gals, gal_config["type"])

    return gals, upos, vpos, noise, noise_scale, colors, flux_zeropoints
