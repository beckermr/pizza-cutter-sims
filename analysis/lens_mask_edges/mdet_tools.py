import numpy as np
import tqdm
import yaml
from loky import get_reusable_executor
import ngmix
import galsim
import copy
import sys
from metadetect.detect import MEDSifier
from ngmix.gaussmom import GaussMom
from pizza_cutter.slice_utils.symmetrize import symmetrize_bmask
from pizza_cutter.slice_utils.interpolate import interpolate_image_and_noise


CONFIG = yaml.safe_load("""\
  metacal:
    psf: fitgauss
    types: [noshear, 1p, 1m, 2p, 2m]
    use_noise_image: True

  psf:
    lm_pars:
      maxfev: 2000
      ftol: 1.0e-05
      xtol: 1.0e-05
    model: gauss

    # we try many times because if this fails we get no psf info
    # for the entire patch
    ntry: 10

  sx:

  weight:
    fwhm: 1.2  # arcsec

  meds:
    box_padding: 2
    box_type: iso_radius
    max_box_size: 53
    min_box_size: 33
    rad_fac: 2
    rad_min: 4

  # check for an edge hit
  bmask_flags: 1610612736  # 2**29 || 2**30

""")


def symmetrize_bmask_nfold(*, bmask, nfolds):
    """symmetrize a bit mask to have N-fold rotational symmetry.

    Parameters
    ----------
    bmask : array-like
        The bit mask.
    nfolds : int
        The desired number of folds in rotational symmetry.

    Returns
    -------
    sym_bmask : array-like
        The symmetrized bit mask
    """
    sym_bmask = bmask.copy()
    if nfolds == 1:
        sym_bmask |= np.rot90(sym_bmask)
    else:
        angles = np.arange(nfolds)[1:] * 360/nfolds
        for angle in angles:
            bmask_rot = bmask.copy()
            symmetrize_bmask(
                bmask=bmask_rot,
                angle=angle,
            )
            sym_bmask |= bmask_rot

    return sym_bmask


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


def _run_boostrap(x1, y1, x2, y2, wgts, verbose):
    rng = np.random.RandomState(seed=100)
    mvals = []
    cvals = []
    if verbose:
        itrl = tqdm.trange(500, leave=False, desc='running bootstrap', ncols=79)
    else:
        itrl = range(500)
    for _ in itrl:
        ind = rng.choice(len(y1), replace=True, size=len(y1))
        _wgts = wgts[ind].copy()
        _wgts /= np.sum(_wgts)
        mvals.append(np.mean(y1[ind] * _wgts) / np.mean(x1[ind] * _wgts) - 1)
        cvals.append(np.mean(y2[ind] * _wgts) / np.mean(x2[ind] * _wgts))

    return (
        np.mean(y1 * wgts) / np.mean(x1 * wgts) - 1, np.std(mvals),
        np.mean(y2 * wgts) / np.mean(x2 * wgts), np.std(cvals))


def _run_jackknife(x1, y1, x2, y2, wgts, jackknife, verbose):
    n_per = x1.shape[0] // jackknife
    n = n_per * jackknife
    x1j = np.zeros(jackknife)
    y1j = np.zeros(jackknife)
    x2j = np.zeros(jackknife)
    y2j = np.zeros(jackknife)
    wgtsj = np.zeros(jackknife)

    loc = 0
    if verbose:
        itrl = tqdm.trange(
            jackknife, desc='running jackknife sums', leave=False, ncols=79
        )
    else:
        itrl = range(jackknife)
    for i in itrl:
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
    if verbose:
        itrl = tqdm.trange(
            jackknife, desc='running jackknife estimates', leave=False, ncols=79
        )
    else:
        itrl = range(jackknife)
    for i in itrl:
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
        np.sqrt((n - n_per) / n * np.sum((mvals-mbar)**2)),
        cbar,
        np.sqrt((n - n_per) / n * np.sum((cvals-cbar)**2)),
    )


def estimate_m_and_c(
    presults,
    mresults,
    g_true,
    swap12=False,
    step=0.01,
    weights=None,
    jackknife=None,
    verbose=False,
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
    verbose : bool, optional
        If True, print progress. Default is False.

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
        return _run_jackknife(x1, y1, x2, y2, wgts, jackknife, verbose)
    else:
        return _run_boostrap(x1, y1, x2, y2, wgts, verbose)


def make_obs(
    *,
    n_grid=6,
    dim=235,
    buff=20,
    scale=0.2,
    psf_fwhm=0.9,
    hlr=0.5,
    nse=1e-7,
    star_dxdy=117,
    star_rad=1,
    n_stars=5,
    seed=10,
    shear=(0.02, 0.0),
    mcal_shear=(0.0, 0.0)
):
    rng = np.random.RandomState(seed=seed)
    n_gals = n_grid**2
    tot_dim = dim + 2*buff
    tot_cen = (tot_dim-1)/2
    gloc = (np.arange(n_grid) + 0.5) * (dim / n_grid) - dim/2
    gloc *= scale
    dx, dy = np.meshgrid(gloc, gloc)
    dx = dx.ravel() + rng.uniform(low=-0.5, high=0.5, size=n_gals) * scale
    dy = dy.ravel() + rng.uniform(low=-0.5, high=0.5, size=n_gals) * scale
    ds = np.arange(n_gals) / (n_gals-1) * 0 + 1
    gals = galsim.Sum([
        galsim.Exponential(
            half_light_radius=hlr * _ds
        ).shift(
            _dx, _dy
        ).shear(
            g1=shear[0], g2=shear[1]
        ).shear(
            g1=mcal_shear[0], g2=mcal_shear[1]
        )
        for _ds, _dx, _dy in zip(ds, dx, dy)
    ])
    psf = galsim.Gaussian(fwhm=psf_fwhm)
    objs = galsim.Convolve([gals, psf])
    im = objs.drawImage(nx=tot_dim, ny=tot_dim, scale=scale).array

    im += rng.normal(size=im.shape, scale=nse)
    nim = rng.normal(size=im.shape, scale=nse)

    psf_dim = 53
    psf_cen = (psf_dim-1)/2
    psf_im = psf.drawImage(nx=psf_dim, ny=psf_dim, scale=scale).array

    # make bmask
    bmask = np.zeros_like(im, dtype=np.int32)
    x, y = np.meshgrid(np.arange(tot_dim), np.arange(tot_dim))
    sdata = []
    for _ in range(n_stars):
        sr2 = np.power(10.0, rng.uniform(low=star_rad, high=star_rad+0.2))**2
        sx = rng.uniform(low=-star_dxdy, high=star_dxdy) + tot_cen
        sy = rng.uniform(low=-star_dxdy, high=star_dxdy) + tot_cen
        dr2 = (x - sx)**2 + (y - sy)**2
        msk = dr2 < sr2
        bmask[msk] |= 2**0
        im[msk] = 0
        sdata.append((sx, sy, np.sqrt(sr2)))

    psf_obs = ngmix.Observation(
        image=psf_im,
        weight=np.ones_like(psf_im) / nse**2,
        jacobian=ngmix.DiagonalJacobian(scale=scale, row=psf_cen, col=psf_cen)
    )
    wgt = np.ones_like(im) / nse**2
    msk = bmask != 0
    wgt[msk] = 0.0
    mfrac = np.zeros_like(im)
    mfrac[msk] = 1.0
    obs = ngmix.Observation(
        image=im,
        noise=nim,
        weight=wgt,
        bmask=bmask,
        ormask=bmask,
        jacobian=ngmix.DiagonalJacobian(scale=scale, row=tot_cen, col=tot_cen),
        psf=psf_obs
    )
    obs.mfrac = mfrac
    mbobs = ngmix.MultiBandObsList()
    obsl = ngmix.ObsList()
    obsl.append(obs)
    mbobs.append(obsl)
    mbobs.meta["sdata"] = sdata
    return mbobs


def meas_mbmeds(mbobs, *, mask_width, sym, maskflags=1, meds_config=None):
    # meas PSF
    mom = GaussMom(fwhm=1.2)
    res = mom.go(obs=mbobs[0][0].psf)
    psf_T = res['T']

    if meds_config is None:
        meds_config = copy.deepcopy(CONFIG["meds"])
    mfier = MEDSifier(
        mbobs,
        sx_config=None,
        meds_config=meds_config,
        maskflags=maskflags
    )
    mbmeds = mfier.get_multiband_meds()
    d = []
    dt = [
        ("flags", "i4"),
        ("g1", "f8"),
        ("g2", "f8"),
        ("s2n", "f8"),
        ("x", "f8"),
        ("y", "f8"),
        ('T_ratio', 'f8'),
    ]
    mw = mask_width
    for i, _mbobs in enumerate(mbmeds.get_mbobs_list()):
        if len(_mbobs) > 0 and len(_mbobs[0]) > 0:
            obs = _mbobs[0][0]
            cen = int((obs.bmask.shape[0]-1)/2)
            if np.any((obs.bmask[cen-mw:cen+mw+1, cen-mw:cen+mw+1] & maskflags) != 0):
                continue

            if sym:
                bmask = obs.bmask.copy()
                if isinstance(sym, list):
                    for angle in sym:
                        bmask_rot = obs.bmask.copy()
                        symmetrize_bmask(
                            bmask=bmask_rot,
                            angle=angle,
                        )
                        bmask |= bmask_rot
                elif isinstance(sym, int):
                    if sym in [2, 4, 8]:
                        angles = np.arange(sym)[1:] * 360/sym
                        for angle in angles:
                            bmask_rot = obs.bmask.copy()
                            symmetrize_bmask(
                                bmask=bmask_rot,
                                angle=angle,
                            )
                            bmask |= bmask_rot
                    else:
                        bmask_rot = obs.bmask.copy()
                        for _ in range(sym):
                            bmask_rot = np.rot90(bmask_rot)
                            bmask |= bmask_rot

                msk = (bmask & maskflags) != 0
                wgt = obs.weight.copy()
                wgt[msk] = 0
                obs.bmask = bmask
                obs.weight = wgt

                if False:
                    import matplotlib.pyplot as plt
                    plt.figure()
                    plt.imshow(obs.weight)
                    import pdb
                    pdb.set_trace()

            mom = GaussMom(fwhm=1.2)
            res = mom.go(obs=obs)
            if res["flags"] == 0:
                d.append((
                    res["flags"],
                    res["e"][0], res["e"][1],
                    res["s2n"],
                    mfier.cat["x"][i], mfier.cat["y"][i],
                    res['T'] / psf_T
                ))
            else:
                d.append((
                    res["flags"],
                    -9999, -9999,
                    -9999,
                    mfier.cat["x"][i], mfier.cat["y"][i],
                    -9999,
                ))
    return np.array(d, dtype=dt), mbobs


def _cut_cat(d):
    return d[
        (d["flags"] == 0)
        & (d["s2n"] > 1e4)
        & (d["T_ratio"] > 1.2)
        & (np.abs(d["g1"]) < 1.0)
        & (np.abs(d["g2"]) < 1.0)
        & ((d["g1"]**2 + d["g2"]**2) < 1.0)
    ]


def _run_one_shear(*, shear, mask_width, sym, **kwargs):
    step = 0.01
    _d, mbobs = meas_mbmeds(
        make_obs(shear=shear, mcal_shear=(0, 0), **kwargs),
        mask_width=mask_width, sym=sym,
    )
    _d1p, mbobs1p = meas_mbmeds(
        make_obs(shear=shear, mcal_shear=(step, 0), **kwargs),
        mask_width=mask_width, sym=sym,
    )
    _d1m, mbobs1m = meas_mbmeds(
        make_obs(shear=shear, mcal_shear=(-step, 0), **kwargs),
        mask_width=mask_width, sym=sym,
    )
    _d2p, mbobs1p = meas_mbmeds(
        make_obs(shear=shear, mcal_shear=(0, step), **kwargs),
        mask_width=mask_width, sym=sym,
    )
    _d2m, mbobs1m = meas_mbmeds(
        make_obs(shear=shear, mcal_shear=(0, -step), **kwargs),
        mask_width=mask_width, sym=sym,
    )
    _d = _cut_cat(_d)
    _d1p = _cut_cat(_d1p)
    _d1m = _cut_cat(_d1m)
    _d2p = _cut_cat(_d2p)
    _d2m = _cut_cat(_d2m)

    if (
        len(_d) > 0
        and len(_d1p) > 0 and len(_d1m) > 0
        and len(_d2p) > 0 and len(_d2m) > 0
    ):
        g1 = np.mean(_d["g1"])
        g1p = np.mean(_d1p["g1"])
        g1m = np.mean(_d1m["g1"])
        g2 = np.mean(_d["g2"])
        g2p = np.mean(_d2p["g2"])
        g2m = np.mean(_d2m["g2"])
        return g1p, g1m, g1, g2p, g2m, g2
    else:
        return None


def _meas_m(*, mask_width, sym, **kwargs):
    pres = _run_one_shear(
        shear=(0.02, 0),
        mask_width=mask_width,
        sym=sym,
        **kwargs,
    )
    if pres is None:
        return None, None, None, None

    mres = _run_one_shear(
        shear=(-0.02, 0),
        mask_width=mask_width,
        sym=sym,
        **kwargs,
    )
    if mres is None:
        return None, None, None, None

    spres = _run_one_shear(
        shear=(0, 0.02),
        mask_width=mask_width,
        sym=sym,
        **kwargs,
    )
    if spres is None:
        return None, None, None, None

    smres = _run_one_shear(
        shear=(0, -0.02),
        mask_width=mask_width,
        sym=sym,
        **kwargs,
    )
    if smres is None:
        return None, None, None, None

    return pres, mres, spres, smres


def meas_m(*, mask_width, sym, n_stars, n_jobs, seed, n_print=500):
    seeds = np.random.RandomState(seed=seed).randint(size=n_jobs, low=1, high=2**28)

    if n_jobs == 1:
        _meas_m(n_stars=n_stars, seed=seed, mask_width=mask_width, sym=sym)
    else:
        exe = get_reusable_executor()
        futs = [
            exe.submit(_meas_m, n_stars=n_stars, seed=s, mask_width=mask_width, sym=sym)
            for s in seeds
        ]
        pres = []
        mres = []
        spres = []
        smres = []
        n_done = 0
        with tqdm.tqdm(
            futs, total=len(futs), ncols=79, file=sys.stdout,
            desc="running sims",
        ) as itrl:
            for fut in itrl:
                n_done += 1
                try:
                    res = fut.result()
                    pres.append(res[0])
                    mres.append(res[1])
                    spres.append(res[2])
                    smres.append(res[3])
                except Exception as e:
                    print(e)

                if n_done % n_print == 0:
                    m, merr, c, cerr = estimate_m_and_c(
                        pres,
                        mres,
                        0.02,
                        jackknife=200 if n_done > 1000 else None,
                    )
                    mstr = "m1 +/- merr: %0.6f +/- %0.6f [10^(-3), 3sigma]" % (
                        m/1e-3, 3*merr/1e-3)
                    itrl.write(mstr, file=sys.stdout)

                    cstr = "c2 +/- cerr: %0.6f +/- %0.6f [10^(-5), 3sigma]" % (
                        c/1e-3, 3*cerr/1e-3)
                    itrl.write(cstr, file=sys.stdout)

                    m, merr, c, cerr = estimate_m_and_c(
                        spres,
                        smres,
                        0.02,
                        jackknife=200 if n_done > 1000 else None,
                        swap12=True,
                    )
                    mstr = "m2 +/- merr: %0.6f +/- %0.6f [10^(-3), 3sigma]" % (
                        m/1e-3, 3*merr/1e-3)
                    itrl.write(mstr, file=sys.stdout)

                    cstr = "c1 +/- cerr: %0.6f +/- %0.6f [10^(-5), 3sigma]" % (
                        c/1e-3, 3*cerr/1e-3)
                    itrl.write(cstr, file=sys.stdout)
                    sys.stdout.flush()

        m1, m1err, c2, c2err = estimate_m_and_c(
            pres,
            mres,
            0.02,
            jackknife=200 if n_jobs > 1000 else None,
        )

        m2, m2err, c1, c1err = estimate_m_and_c(
            spres,
            smres,
            0.02,
            jackknife=200 if n_jobs > 1000 else None,
            swap12=True,
        )

        return dict(
            m1=m1,
            m1err=m1err,
            c2=c2,
            c2err=c2err,
            pres=pres,
            mres=mres,
            m2=m2,
            m2err=m2err,
            c1=c1,
            c1err=c1err,
            spres=spres,
            smres=smres,
        )


def format_mc_res(res, space=None):
    fstrs = []
    m, merr = res["m1"], res["m1err"]
    fstrs.append("m1 +/- merr: %0.6f +/- %0.6f [10^(-3), 3sigma]" % (
        m/1e-3, 3*merr/1e-3
    ))
    m, merr = res["m2"], res["m2err"]
    fstrs.append("m2 +/- merr: %0.6f +/- %0.6f [10^(-3), 3sigma]" % (
        m/1e-3, 3*merr/1e-3
    ))

    c, cerr = res["c1"], res["c1err"]
    fstrs.append("c1 +/- cerr: %0.6f +/- %0.6f [10^(-5), 3sigma]" % (
        c/1e-3, 3*cerr/1e-3
    ))

    c, cerr = res["c2"], res["c2err"]
    fstrs.append("c2 +/- cerr: %0.6f +/- %0.6f [10^(-5), 3sigma]" % (
        c/1e-3, 3*cerr/1e-3
    ))

    if space is not None and space > 0:
        st = "\n" + (" " * space)
    else:
        st = "\n"

    return st.join(fstrs)


def meas_one_im(*, g1, g2, seed, n_stars=0, sym_nfold=None, interp=False):
    rng = np.random.RandomState(seed=seed)

    obj = galsim.Exponential(half_light_radius=0.5).shear(g1=g1, g2=g2)
    psf = galsim.Gaussian(fwhm=0.9).withFlux(1e6)
    obj = galsim.Convolve([obj, psf])
    dim = 53
    cen = (dim-1)//2
    offset = rng.uniform(low=-0.5, high=0.5, size=2)
    im = obj.drawImage(nx=dim, ny=dim, scale=0.2, offset=offset).array
    jac = jac = ngmix.DiagonalJacobian(
        scale=0.2,
        row=cen+offset[1],
        col=cen+offset[0],
    )
    psf_im = psf.drawImage(nx=dim, ny=dim, scale=0.2).array
    psf_jac = ngmix.DiagonalJacobian(scale=0.2, row=cen, col=cen)
    psf_obs = ngmix.Observation(
        image=psf_im,
        weight=np.ones_like(psf_im),
        jacobian=psf_jac,
    )
    wgt = np.ones_like(im)

    bmask = np.zeros_like(im, dtype=np.int32)

    if True:
        for _ in range(n_stars):
            srad = np.power(10, rng.uniform(low=1, high=3))
            ang = rng.uniform(low=0, high=2.0*np.pi)
            lrad = rng.uniform(low=(srad - (dim/2-1)), high=srad-(dim/2-10)) + dim/2
            xc = lrad * np.cos(ang) + dim/2
            yc = lrad * np.sin(ang) + dim/2
            srad2 = srad * srad
            x, y = np.meshgrid(np.arange(dim), np.arange(dim))
            msk = ((x-xc)**2 + (y-yc)**2) < srad2
            bmask[msk] = 1
    else:
        import scipy.ndimage
        msk = np.zeros_like(bmask)
        angle = rng.uniform(low=0, high=360) * 0
        col = int(rng.uniform(low=dim/2, high=dim-1))
        msk[:, col:] = 1
        msk = scipy.ndimage.rotate(
            msk,
            angle,
            reshape=False,
            order=1,
            mode='constant',
            cval=1.0,
        )
        bmask[msk == 1] = 1
    if sym_nfold is not None:
        bmask = symmetrize_bmask_nfold(bmask=bmask, nfolds=sym_nfold)

    msk = bmask != 0
    wgt[msk] = 0
    im[msk] = np.nan

    if interp:
        nse = rng.normal(size=im.shape)
        iim, inse = interpolate_image_and_noise(
            image=im,
            noises=[nse],
            weight=wgt,
            bmask=bmask,
            bad_flags=1,
            rng=rng,
            fill_isolated_with_noise=True
        )

    obs = ngmix.Observation(
        image=im,
        weight=wgt,
        jacobian=jac,
        psf=psf_obs,
        bmask=bmask,
    )
    mom = GaussMom(fwhm=1.2)
    res = mom.go(obs=obs)

    gauss_wgt = ngmix.GMixModel(
        [0, 0, 0, 0, ngmix.moments.fwhm_to_T(1.2), 1],
        'gauss',
    )
    cobs = obs.copy()
    cobs.image = 1.0 - wgt
    cobs.weight = np.ones_like(wgt)
    stats = gauss_wgt.get_weighted_sums(
        cobs,
        1.2 * 2,
    )
    mfrac = stats["sums"][5] / stats["wsum"]
    return res["e"][0], res["e"][1], obs, mfrac


def meas_response_one_im(seed, n_stars=0, sym_nfold=None, interp=False, swap12=False):
    e1 = []
    e2 = []
    shear = np.linspace(0, 0.06, 50)
    for s in shear:
        if swap12:
            _g1 = 0
            _g2 = s
        else:
            _g1 = s
            _g2 = 0
        _e1, _e2, _obs, _mfrac = meas_one_im(
            g1=_g1, g2=_g2, seed=seed, sym_nfold=sym_nfold, interp=interp,
            n_stars=n_stars,
        )
        e1.append(_e1)
        e2.append(_e2)
    e1 = np.array(e1)
    e2 = np.array(e2)
    if swap12:
        R = (e2[1:] - e2[:-1])/(shear[1:] - shear[:-1])
    else:
        R = (e1[1:] - e1[:-1])/(shear[1:] - shear[:-1])
    sp = (shear[1:] + shear[:-1])/2
    ds = sp[1] - sp[0]
    if swap12:
        _g1 = 0
        _g2 = ds
        ind = 1
    else:
        _g1 = ds
        _g2 = 0
        ind = 0
    R0 = (
        meas_one_im(
            g1=_g1, g2=_g2, seed=seed, sym_nfold=sym_nfold, interp=interp,
            n_stars=n_stars,
        )[ind]
        - meas_one_im(
            g1=-_g1, g2=-_g2, seed=seed, sym_nfold=sym_nfold, interp=interp,
            n_stars=n_stars,
        )[ind]
    )/2/ds
    sp = np.concatenate([np.array([0]), sp])
    R = np.concatenate([np.array([R0]), R])
    mfrac = meas_one_im(
        g1=0, g2=0, seed=seed, sym_nfold=sym_nfold, interp=interp,
        n_stars=n_stars,
    )[-1]

    return dict(
        shear=sp,
        R=R,
        mfrac=mfrac,
        e1=e1,
        e2=e2,
        obs=_obs,
    )
