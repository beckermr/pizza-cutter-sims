import copy
import galsim


def build_gsobject(*, config, kind, rng=None, eval_variables=None):
    """Build a galsim object.

    Parameters
    ----------
    config : dict
        A galsim-style config for the object.
    kind : str
        One of 'gal' or 'psf'. This parameter effects how galsim interprets
        the object.
    rng : galsim.BaseDeviate or None, optional.
        This is an optional RNG to allow the config to specify random properties
        for the object.
    eval_variables : dict or None, optional
        This is an optional dictionary with variables to be used in Eval-type
        galsim config specifications.

    Returns
    -------
    obj : galsim.GSObject
        The galsim object.
    """
    top = {
        "rng": rng,
        kind: copy.deepcopy(config),
        "eval_variables": eval_variables,
    }
    obj, _ = galsim.config.BuildGSObject(top, kind)
    return obj
