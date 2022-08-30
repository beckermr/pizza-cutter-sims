import multiprocessing
import contextlib
import time
from contextlib import contextmanager
import sys

import numpy as np
import tqdm

from mattspy import SLACLSFParallel, LokyParallel, BNLCondorParallel

GLOBAL_START_TIME = time.time()


@contextmanager
def timer(name, silent=False):
    t0 = time.time()
    if not silent:
        print(
            "[% 8ds] %s" % (t0 - GLOBAL_START_TIME, name),
            flush=True,
            file=sys.stderr,
        )
    yield
    t1 = time.time()
    if not silent:
        print(
            "[% 8ds] %s done (%f seconds)" % (
                t1 - GLOBAL_START_TIME,
                name,
                t1 - t0
            ),
            flush=True,
            file=sys.stderr,
        )


def get_n_workers(backend, n_workers=None):
    """Get the number of workers.

    Parameters
    ----------
    backend : str
        One of 'sequential', `dask`, `multiprocessing`, `loky`, or 'mpi'.
    n_workers : int, optional
        The number of workers to use. Defaults to 1 for the 'sequential' backend,
        the cpu count for the 'loky' backend, and the size of the default global
        communicator for the 'mpi' backend.

    Returns
    -------
    n_workers : int
        The number of workers
    """
    if backend == "loky":
        return n_workers or multiprocessing.cpu_count()
    elif backend == "mpi":
        from mpi4py import MPI
        return MPI.COMM_WORLD.Get_size()
    elif backend == "local":
        return 1
    elif backend == "condor":
        return 10000
    elif backend == "lsf":
        return 1000
    else:
        raise RuntimeError("backend '%s' not recognized!" % backend)


@contextlib.contextmanager
def backend_pool(backend, n_workers=None, verbose=0, **kwargs):
    """Context manager to build a schwimmbad `pool` object with the `map` method.

    Parameters
    ----------
    backend : str
        One of 'condor', 'lsf', 'loky', or 'mpi'.
    n_workers : int, optional
        The number of workers to use. Defaults to 1 for the 'sequential' backend,
        the cpu count for the 'loky' backend, and the size of the default global
        communicator for the 'mpi' backend.
    verbose : int
        Higher values print/save more output.
    **kwargs : extra keyword arguments
        Passed to the backend.
    """

    local_env_overrides = {
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "VECLIB_MAXIMUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
    }

    if backend == "lsf":
        with SLACLSFParallel(
            verbose=100, n_jobs=n_workers, **kwargs
        ) as pool:
            yield pool
    elif backend == "condor":
        with BNLCondorParallel(
            verbose=verbose, n_jobs=n_workers, **kwargs
        ) as pool:
            yield pool
    elif backend == "loky":
        _n_workers = get_n_workers(backend, n_workers=n_workers)
        with LokyParallel(
            n_jobs=_n_workers,
            env=local_env_overrides,
            verbose=verbose,
            **kwargs,
        ) as pool:
            yield pool
    else:
        raise RuntimeError("backend '%s' not recognized!" % backend)


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


def _run_boostrap(x1, y1, x2, y2, wgts, silent):
    rng = np.random.RandomState(seed=100)
    mvals = []
    cvals = []
    if silent:
        itrl = range(500)
    else:
        itrl = tqdm.trange(
            500, leave=False, desc='running bootstrap', ncols=79,
            file=sys.stderr,
        )
    for _ in itrl:
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
    n = n_per * jackknife
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

    # weighted jackknife from Busing et al. 1999, Statistics and Computing, 9, 3-8
    mhat = np.mean(y1[:n] * wgts[:n]) / np.mean(x1[:n] * wgts[:n]) - 1
    chat = np.mean(y2[:n] * wgts[:n]) / np.mean(x2[:n] * wgts[:n])
    mhatj = np.zeros(jackknife)
    chatj = np.zeros(jackknife)
    for i in range(jackknife):
        _wgts = np.delete(wgtsj, i)
        mhatj[i] = (
            np.sum(np.delete(y1j, i) * _wgts) / np.sum(np.delete(x1j, i) * _wgts)
            - 1
        )
        chatj[i] = (
            np.sum(np.delete(y2j, i) * _wgts) / np.sum(np.delete(x2j, i) * _wgts)
        )

    tot_wgt = np.sum(wgtsj)
    mbar = jackknife * mhat - np.sum((1.0 - wgtsj/tot_wgt) * mhatj)
    cbar = jackknife * chat - np.sum((1.0 - wgtsj/tot_wgt) * chatj)

    hj = tot_wgt / wgtsj
    mtildej = hj * mhat - (hj - 1) * mhatj
    ctildej = hj * chat - (hj - 1) * chatj

    mvarj = np.sum((mtildej - mbar)**2 / (hj-1)) / jackknife
    cvarj = np.sum((ctildej - cbar)**2 / (hj-1)) / jackknife

    return (
        mbar,
        np.sqrt(mvarj),
        cbar,
        np.sqrt(cvarj),
    )


def estimate_m_and_c(
    presults,
    mresults,
    g_true,
    swap12=False,
    step=0.01,
    weights=None,
    jackknife=None,
    silent=False,
):
    """Estimate m and c from paired lensing simulations.

    Parameters
    ----------
    presults : list of iterables or np.ndarray
        A list of iterables, each with g1p, g1m, g1, g2p, g2m, g2
        from running metadetect with a `g1` shear in the 1-component and
        0 true shear in the 2-component. If an array, it should have the named
        columns.
    mresults : list of iterables or np.ndarray
        A list of iterables, each with g1p, g1m, g1, g2p, g2m, g2
        from running metadetect with a -`g1` shear in the 1-component and
        0 true shear in the 2-component. If an array, it should have the named
        columns.
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
    silent : bool, optional
        If True, do not print to stderr/stdout.

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

    with timer("prepping data for m,c measurement", silent=silent):
        if isinstance(presults, list) or isinstance(mresults, list):
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
        else:
            if swap12:
                g1p = presults["g2"]
                R11p = (presults["g2p"] - presults["g2m"]) / 2 / step * g_true
                g2p = presults["g1"]
                R22p = (presults["g1p"] - presults["g1m"]) / 2 / step

                g1m = mresults["g2"]
                R11m = (mresults["g2p"] - mresults["g2m"]) / 2 / step * g_true
                g2m = mresults["g1"]
                R22m = (mresults["g1p"] - mresults["g1m"]) / 2 / step
            else:
                g1p = presults["g1"]
                R11p = (presults["g1p"] - presults["g1m"]) / 2 / step * g_true
                g2p = presults["g2"]
                R22p = (presults["g2p"] - presults["g2m"]) / 2 / step

                g1m = mresults["g1"]
                R11m = (mresults["g1p"] - mresults["g1m"]) / 2 / step * g_true
                g2m = mresults["g2"]
                R22m = (mresults["g2p"] - mresults["g2m"]) / 2 / step

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
        with timer("running jackknife", silent=silent):
            return _run_jackknife(x1, y1, x2, y2, wgts, jackknife)
    else:
        with timer("running bootstrap", silent=silent):
            return _run_boostrap(x1, y1, x2, y2, wgts, silent)


def measure_shear_metadetect(
    res, *, s2n_cut, t_ratio_cut, ormask_cut, mfrac_cut, model
):
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
    ormask_cut : bool
        If True, cut on the `ormask` flags.
    mfrac_cut : float or None
        If not None, cut objects with a masked fraction higher than this
        value.
    model : str
        The model kind (e.g. wmom).

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
        if 'flags' in data.dtype.names:
            flag_col = "flags"
        else:
            flag_col = model + "_flags"

        _cut_msk = (
            (data[flag_col] == 0)
            & (data[model + '_s2n'] > s2n_cut)
            & (data[model + '_T_ratio'] > t_ratio_cut)
        )
        if ormask_cut:
            _cut_msk = _cut_msk & (data['ormask'] == 0)
        if mfrac_cut is not None:
            _cut_msk = _cut_msk & (data["mfrac"] <= mfrac_cut)
        return _cut_msk

    op = res['1p']
    q = _mask(op)
    if not np.any(q):
        return None
    g1p = op[model + '_g'][q, 0]

    om = res['1m']
    q = _mask(om)
    if not np.any(q):
        return None
    g1m = om[model + '_g'][q, 0]

    o = res['noshear']
    q = _mask(o)
    if not np.any(q):
        return None
    g1 = o[model + '_g'][q, 0]
    g2 = o[model + '_g'][q, 1]

    op = res['2p']
    q = _mask(op)
    if not np.any(q):
        return None
    g2p = op[model + '_g'][q, 1]

    om = res['2m']
    q = _mask(om)
    if not np.any(q):
        return None
    g2m = om[model + '_g'][q, 1]

    return (
        np.mean(g1p), np.mean(g1m), np.mean(g1),
        np.mean(g2p), np.mean(g2m), np.mean(g2))
