import numpy as np


def make_glass_layout(n_gals, pos_bounds, rng):
    """Make a glass-like layout.

    Note this function may return fewer than the requested number of points.

    Parameters
    ----------
    n_gals : int
        The number of objects to draw.
    pos_bounds : 2-tuple of floats
        The range in which to draw the objects.
    rng : np.random.RandomState
        An RNG object to use to draw positions.

    Returns
    -------
    u : array-like
        The x/column positions.
    v : array-like
        The y/row positions.
    """
    upos = np.zeros(n_gals)
    vpos = np.zeros(n_gals)
    delta = (pos_bounds[1] - pos_bounds[0]) / np.sqrt(n_gals)
    delta2 = delta**2

    loc = 0
    tries = 0
    max_tries = n_gals * 200
    while loc < n_gals and tries < max_tries:
        tries += 1
        u = rng.uniform(low=pos_bounds[0], high=pos_bounds[1])
        v = rng.uniform(low=pos_bounds[0], high=pos_bounds[1])
        dr2 = (upos[:loc] - u)**2 + (vpos[:loc] - v)**2
        if loc == 0 or np.min(dr2) >= delta2:
            upos[loc] = u
            vpos[loc] = v
            loc += 1

    return upos[:loc], vpos[:loc]
