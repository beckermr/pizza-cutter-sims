import numpy as np
from numba import njit

FIDUCIAL_COADD_AREA = (0.263*250)**2


def generate_cosmic_rays(
    shape, mean_cosmic_rays=10,
    min_length=10, max_length=30, rng=None, area=None
):
    """Generate a binary mask w/ cosmic rays.

    This routine generates cosmic rays by choosing a random
    position in the image, a random angle, and a random
    length of the cosmic ray track. Then pixels along the line
    determined by the length of the track, the position, and the angle
    are marked. The width of the track is broadened a bit when
    the row or column changes.

    The total number of cosmic rays is Poisson distributed according to
    `mean_cosmic_rays`.

    The length of the track is chosen uniformly between `min_length` and
    `max_length`.

    Parameters
    ----------
    shape : int or tuple of ints
        The shape of the mask to generate.
    mean_cosmic_rays : int, optional
        The mean number of cosmic rays.
    min_length : int, optional
        The minimum length of the track.
    max_length : int, optional
        The maximum length of the track.
    rng : np.random.RandomState or None, optional
        An RNG to use. If none is provided, a new `np.random.RandomState`
        state instance will be created.
    area : float
        If not None, scale the mean rate by area / FIDUCIAL_COADD_AREA.

    Returns
    -------
    msk : np.ndarray, shape `shape`
        A boolean mask marking the locations of the cosmic rays.
    """
    msk = np.zeros(shape)
    rng = rng or np.random.RandomState()
    if area is not None:
        mean_cosmic_rays *= (area / FIDUCIAL_COADD_AREA)
    n_cosmic_rays = rng.poisson(mean_cosmic_rays)

    for _ in range(n_cosmic_rays):
        y = rng.randint(0, msk.shape[0]-1)
        x = rng.randint(0, msk.shape[1]-1)
        angle = rng.uniform() * 2.0 * np.pi
        n_pix = rng.randint(min_length, max_length+1)
        cosa = np.cos(angle)
        sina = np.sin(angle)

        _x_prev = None
        _y_prev = None
        for _ in range(n_pix):
            _x = int(x + 0.5)
            _y = int(y + 0.5)
            if _y >= 0 and _y < msk.shape[0] and _x >= 0 and _x < msk.shape[1]:
                if _x_prev is not None and _y_prev is not None:
                    if _x_prev != _x:
                        msk[_y, _x_prev] = 1
                    if _y_prev != _y:
                        msk[_y_prev, _x] = 1
                msk[_y, _x] = 1
                _x_prev = _x
                _y_prev = _y
            x += cosa
            y += sina

    return msk.astype(bool)


def generate_bad_columns(
    shape, mean_bad_cols=10,
    widths=(1, 2, 5, 10), p=(0.8, 0.1, 0.075, 0.025),
    min_length_frac=(1, 1, 0.25, 0.25),
    max_length_frac=(1, 1, 0.75, 0.75),
    gap_prob=(0.30, 0.30, 0, 0),
    min_gap_frac=(0.1, 0.1, 0, 0),
    max_gap_frac=(0.3, 0.3, 0, 0),
    rng=None, area=None,
):
    """Generate a binary mask w/ bad columns.

    Parameters
    ----------
    shape : int or tuple of ints
        The shape of the mask to generate.
    mean_bad_cols : float, optional
        The mean of the Poisson distribution for the total number of
        bad columns to generate.
    widths : n-tuple of ints, optional
        The possible widths of the bad columns.
    p : n-tuple of floats, optional
        The frequency of each of the bad column widths.
    min_length_frac : n-tuple of floats, optional
        The minimum fraction of the image the bad column spans. There should be
        one entry per bad column width in `widths`.
    max_length_frac : n-tuple of floats, optional
        The maximum fraction of the image the bad column spans. There should be
        one entry per bad column width in `widths`.
    gap_prob : n-tuple of floats, optional
        The probability that the bad column has a gap in it. There should be
        one entry per bad column width in `widths`.
    min_gap_frac : n-tuple of floats, optional
        The minimum fraction of the image that the gap spans. There should be
        one entry per bad column width in `widths`.
    max_gap_frac : n-tuple of floats, optional
        The maximum fraction of the image that the gap spans. There should be
        one entry per bad column width in `widths`.
    rng : np.random.RandomState or None, optional
        An RNG to use. If none is provided, a new `np.random.RandomState`
        state instance will be created.
    area : float
        If not None, scale the mean rate by sqrt(area / FIDUCIAL_COADD_AREA).

    Returns
    -------
    msk : np.ndarray, shape `shape`
        A boolean mask marking the locations of the bad columns.
    """
    p = np.array(p) / np.sum(p)
    msk = np.zeros(shape)
    rng = rng or np.random.RandomState()
    if area is not None:
        mean_bad_cols *= np.sqrt(area / FIDUCIAL_COADD_AREA)
    n_bad_cols = rng.poisson(mean_bad_cols)

    for _ in range(n_bad_cols):
        w = rng.choice(widths, p=p)
        wind = widths.index(w)
        x = rng.randint(0, msk.shape[1]-w)

        len_frac = rng.uniform(min_length_frac[wind], max_length_frac[wind])
        length = int(len_frac * msk.shape[0])

        if length < msk.shape[0]:
            start = rng.choice([0, msk.shape[0] - length])
        else:
            start = 0

        # set the mask first
        msk[start:start+length, x:x+w] = 1

        # add gaps
        if rng.uniform() < gap_prob[wind]:
            gfrac = rng.uniform(min_gap_frac[wind], max_gap_frac[wind])
            glength = int(msk.shape[0] * gfrac)
            gloc = rng.randint(0, msk.shape[0] - glength)
            msk[gloc:gloc+glength, x:x+1] = 0

    return msk.astype(bool)


@njit
def _point_line_dist(x1, y1, x2, y2, x0, y0):
    return np.abs((x2-x1)*(y1-y0) - (x1-x0)*(y2-y1)) / np.sqrt((x2-x1)**2 + (y2-y1)**2)


@njit
def _fill_streaks_mask(msk, shape, x1s, y1s, x2s, y2s, half_widths):
    for yind in range(shape[0]):
        y0 = yind + 0.5
        for xind in range(shape[1]):
            x0 = xind + 0.5
            for x1, y1, x2, y2, half_width in zip(x1s, y1s, x2s, y2s, half_widths):
                if _point_line_dist(x1, y1, x2, y2, x0, y0) < half_width:
                    msk[yind, xind] = 1


def generate_streaks(
    shape, mean_streaks=1, min_wdth=2, max_width=10, rng=None, area=None,
):
    """Make image streaks representing airplanes and other flying things.

    The streak width range is guess based on
    https://iopscience.iop.org/article/10.3847/1538-3881/aaddff

    Parameters
    ----------
    shape : tuple of ints
        The shape of the mask to generate.
    mean_steaks : float, optional
        The mean of the Poisson distribution for the total number of
        streaks to generate.
    min_width : float
        The minimum width of a streak. Default is 2 pixels.
    max_width : float
        The maximum width of a streak. Default is 10 pixels.
    rng : np.random.RandomState or None, optional
        An RNG to use. If none is provided, a new `np.random.RandomState`
        state instance will be created.
    area : float
        If not None, scale the mean rate by area / FIDUCIAL_COADD_AREA.

    Returns
    -------
    msk : np.ndarray, shape `shape`
        A boolean mask marking the locations of the bad columns.
    """
    msk = np.zeros(shape, dtype=np.int32)
    rng = rng or np.random.RandomState()
    if area is not None:
        mean_streaks *= (area / FIDUCIAL_COADD_AREA)

    n_streaks = rng.poisson(mean_streaks)
    if n_streaks > 0:
        x1s = rng.uniform(size=n_streaks) * shape[1]
        y1s = rng.uniform(size=n_streaks) * shape[0]
        angs = rng.uniform(size=n_streaks) * np.pi * 2.0
        x2s = x1s + np.cos(angs)
        y2s = y1s + np.sin(angs)
        half_widths = rng.uniform(
            low=min_wdth,
            high=max_width,
            size=n_streaks,
        ) / 2
        _fill_streaks_mask(msk, shape, x1s, y1s, x2s, y2s, half_widths)

    return msk.astype(bool)
