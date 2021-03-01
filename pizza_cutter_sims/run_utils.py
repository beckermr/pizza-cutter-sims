import multiprocessing
import contextlib

import numpy as np
import tqdm
import schwimmbad


@contextlib.contextmanager
def backend_pool(backend, n_workers=None):
    """Context manager to build a schwimmbad `pool` object with the `map` method.

    Parameters
    ----------
    backend : str
        One of 'sequential', `loky`, or 'mpi'.
    n_workers : int, optional
        The number of workers to use. Defaults to 1 for the 'sequential' backend,
        the cpu count for the 'loky' backend, and the size of the default global
        communicator for the 'mpi' backend.
    """
    try:
        if backend == "sequential":
            pool = schwimmbad.JoblibPool(1, backend=backend, verbose=0)
        else:
            if backend == "mpi":
                from mpi4py import MPI
                pool = schwimmbad.choose_pool(
                    mpi=True,
                    processes=n_workers or MPI.COMM_WORLD.Get_size(),
                )
            else:
                pool = schwimmbad.JoblibPool(
                    n_workers or multiprocessing.cpu_count(),
                    backend=backend,
                    verbose=100,
                )
        yield pool
    finally:
        if "pool" in locals():
            pool.close()


def cut_nones(presults, mresults):
    """Cut entries that are None in a pair of lists. Any entry that is None
    in either list will exclude the item in the other.

    Parameters
    ----------
    presults : list
        One the list of things.
    mresults : list
        The other list of things.

    Returns
    -------
    pcut : list
        The cut list.
    mcut : list
        The cut list.
    """
    prr_keep = []
    mrr_keep = []
    for pr, mr in zip(presults, mresults):
        if pr is None or mr is None:
            continue
        prr_keep.append(pr)
        mrr_keep.append(mr)

    return prr_keep, mrr_keep


def _run_boostrap(x1, y1, x2, y2, wgts):
    rng = np.random.RandomState(seed=100)
    mvals = []
    cvals = []
    for _ in tqdm.trange(500, leave=False):
        ind = rng.choice(len(y1), replace=True, size=len(y1))
        _wgts = wgts[ind].copy()
        _wgts /= np.sum(_wgts)
        mvals.append(np.mean(y1[ind] * _wgts) / np.mean(x1[ind] * _wgts) - 1)
        cvals.append(np.mean(y2[ind] * _wgts) / np.mean(x2[ind] * _wgts))

    return (
        np.mean(y1 * wgts) / np.mean(x1 * wgts) - 1, np.std(mvals),
        np.mean(y2 * wgts) / np.mean(x2 * wgts), np.std(cvals))


def _run_jackknife(x1, y1, x2, y2, wgts, jackknife):
    n_per = x1.shape[0] // jackknife
    x1j = np.zeros(jackknife)
    y1j = np.zeros(jackknife)
    x2j = np.zeros(jackknife)
    y2j = np.zeros(jackknife)
    wgtsj = np.zeros(jackknife)

    loc = 0
    for i in range(jackknife):
        wgtsj[i] = np.sum(wgts[loc:loc+n_per])
        x1j[i] = np.sum(x1[loc:loc+n_per] * wgts[loc:loc+n_per]) / wgtsj[i]
        y1j[i] = np.sum(y1[loc:loc+n_per] * wgts[loc:loc+n_per]) / wgtsj[i]
        x2j[i] = np.sum(x2[loc:loc+n_per] * wgts[loc:loc+n_per]) / wgtsj[i]
        y2j[i] = np.sum(y2[loc:loc+n_per] * wgts[loc:loc+n_per]) / wgtsj[i]

        loc += n_per

    mbar = np.mean(y1 * wgts) / np.mean(x1 * wgts) - 1
    cbar = np.mean(y2 * wgts) / np.mean(x2 * wgts)
    mvals = np.zeros(jackknife)
    cvals = np.zeros(jackknife)
    for i in range(jackknife):
        _wgts = np.delete(wgtsj, i)
        mvals[i] = (
            np.sum(np.delete(y1j, i) * _wgts) / np.sum(np.delete(x1j, i) * _wgts)
            - 1
        )
        cvals[i] = (
            np.sum(np.delete(y2j, i) * _wgts) / np.sum(np.delete(x2j, i) * _wgts)
        )

    return (
        mbar,
        np.sqrt((jackknife - n_per) / jackknife * np.sum((mvals-mbar)**2)),
        cbar,
        np.sqrt((jackknife - n_per) / jackknife * np.sum((cvals-cbar)**2)),
    )


def estimate_m_and_c(
    presults,
    mresults,
    g_true,
    swap12=False,
    step=0.01,
    weights=None,
    jackknife=None,
):
    """Estimate m and c from paired lensing simulations.

    Parameters
    ----------
    presults : list of iterables
        A list of iterables, each with g1p, g1m, g1, g2p, g2m, g2
        from running metadetect with a `g1` shear in the 1-component and
        0 true shear in the 2-component.
    mresults : list of iterables
        A list of iterables, each with g1p, g1m, g1, g2p, g2m, g2
        from running metadetect with a -`g1` shear in the 1-component and
        0 true shear in the 2-component.
    g_true : float
        The true value of the shear on the 1-axis in the simulation. The other
        axis is assumd to havea true value of zero.
    swap12 : bool, optional
        If True, swap the roles of the 1- and 2-axes in the computation.
    step : float, optional
        The step used in metadetect for estimating the response. Default is
        0.01.
    weights : list of weights, optional
        Weights to apply to each sample. Will be normalized if not already.
    jackknife : int, optional
        The number of jackknife sections to use for error estimation. Default of
        None will do no jackknife and default to bootstrap error bars.

    Returns
    -------
    m : float
        Estimate of the multiplicative bias.
    merr : float
        Estimat of the 1-sigma standard error in `m`.
    c : float
        Estimate of the additive bias.
    cerr : float
        Estimate of the 1-sigma standard error in `c`.
    """

    prr_keep, mrr_keep = cut_nones(presults, mresults)

    def _get_stuff(rr):
        _a = np.vstack(rr)
        g1p = _a[:, 0]
        g1m = _a[:, 1]
        g1 = _a[:, 2]
        g2p = _a[:, 3]
        g2m = _a[:, 4]
        g2 = _a[:, 5]

        if swap12:
            g1p, g1m, g1, g2p, g2m, g2 = g2p, g2m, g2, g1p, g1m, g1

        return (
            g1, (g1p - g1m) / 2 / step * g_true,
            g2, (g2p - g2m) / 2 / step)

    g1p, R11p, g2p, R22p = _get_stuff(prr_keep)
    g1m, R11m, g2m, R22m = _get_stuff(mrr_keep)

    if weights is not None:
        wgts = np.array(weights).astype(np.float64)
    else:
        wgts = np.ones(len(g1p)).astype(np.float64)
    wgts /= np.sum(wgts)

    msk = (
        np.isfinite(g1p) &
        np.isfinite(R11p) &
        np.isfinite(g1m) &
        np.isfinite(R11m) &
        np.isfinite(g2p) &
        np.isfinite(R22p) &
        np.isfinite(g2m) &
        np.isfinite(R22m))
    g1p = g1p[msk]
    R11p = R11p[msk]
    g1m = g1m[msk]
    R11m = R11m[msk]
    g2p = g2p[msk]
    R22p = R22p[msk]
    g2m = g2m[msk]
    R22m = R22m[msk]
    wgts = wgts[msk]

    x1 = (R11p + R11m)/2
    y1 = (g1p - g1m) / 2

    x2 = (R22p + R22m) / 2
    y2 = (g2p + g2m) / 2

    if jackknife:
        return _run_jackknife(x1, y1, x2, y2, wgts, jackknife)
    else:
        return _run_boostrap(x1, y1, x2, y2, wgts)


def measure_shear_metadetect(res, *, s2n_cut, t_ratio_cut, cut_interp):
    """Measure the shear parameters for metadetect.

    NOTE: Returns None if nothing can be measured.

    Parameters
    ----------
    res : dict
        The metadetect results.
    s2n_cut : float
        The cut on `wmom_s2n`. Typically 10.
    t_ratio_cut : float
        The cut on `t_ratio_cut`. Typically 1.2.
    cut_interp : bool
        If True, cut on the `ormask` flags.

    Returns
    -------
    g1p : float
        The mean 1-component shape for the plus metadetect measurement.
    g1m : float
        The mean 1-component shape for the minus metadetect measurement.
    g1 : float
        The mean 1-component shape for the zero-shear metadetect measurement.
    g2p : float
        The mean 2-component shape for the plus metadetect measurement.
    g2m : float
        The mean 2-component shape for the minus metadetect measurement.
    g2 : float
        The mean 2-component shape for the zero-shear metadetect measurement.
    """
    def _mask(data):
        if cut_interp:
            return (
                (data['flags'] == 0) &
                (data['ormask'] == 0) &
                (data['wmom_s2n'] > s2n_cut) &
                (data['wmom_T_ratio'] > t_ratio_cut))
        else:
            return (
                (data['flags'] == 0) &
                (data['wmom_s2n'] > s2n_cut) &
                (data['wmom_T_ratio'] > t_ratio_cut))

    op = res['1p']
    q = _mask(op)
    if not np.any(q):
        return None
    g1p = op['wmom_g'][q, 0]

    om = res['1m']
    q = _mask(om)
    if not np.any(q):
        return None
    g1m = om['wmom_g'][q, 0]

    o = res['noshear']
    q = _mask(o)
    if not np.any(q):
        return None
    g1 = o['wmom_g'][q, 0]
    g2 = o['wmom_g'][q, 1]

    op = res['2p']
    q = _mask(op)
    if not np.any(q):
        return None
    g2p = op['wmom_g'][q, 1]

    om = res['2m']
    q = _mask(om)
    if not np.any(q):
        return None
    g2m = om['wmom_g'][q, 1]

    return (
        np.mean(g1p), np.mean(g1m), np.mean(g1),
        np.mean(g2p), np.mean(g2m), np.mean(g2))
