import ngmix
import numpy as np
import galsim
from ..mdet import gen_metadetect_color_dep
from ..psf import GalsimPSF


def test_gen_metadetect_color_dep():
    scale = 0.2
    dim = 275
    coadd_cen = (dim-1)/2
    coadd_cen_pos = galsim.PositionD(coadd_cen, coadd_cen)
    coadd_wcs = galsim.AffineTransform(
        scale,
        0,
        0,
        scale,
        origin=coadd_cen_pos,
    )

    def color_dep_fun(color):
        return 1.0 + color*0.2

    nbands = 3
    mbobs = ngmix.MultiBandObsList()
    psfs = []
    for band in range(nbands):
        psf_obj = galsim.Gaussian(fwhm=band*0.2 + 0.8)
        obslist = ngmix.ObsList()
        obs = ngmix.Observation(
            image=np.zeros((dim, dim)),
            psf=ngmix.Observation(image=np.ones((25, 25))),
        )
        obslist.append(obs)
        mbobs.append(obslist)
        psfs.append(
            GalsimPSF(psf_obj, color_dep_fun)
        )

    color_key_func, color_dep_mbobs = gen_metadetect_color_dep(
        psfs=psfs,
        coadd_wcs=coadd_wcs,
        mbobs=mbobs,
        coadd_cen_pos=coadd_cen_pos,
        color_range=[0, 2],
        ncolors=33,
        flux_zeropoints=[30, 30],
    )
    assert color_key_func([-1, -2]) == 0
    assert color_key_func([1, 1]) == 0
    assert color_key_func([1, 10**(0.4 * 2)]) == 32
    assert color_key_func([-1, -0.5]) == 32

    cols = np.linspace(0, 2, 33)
    cind = int((0.5-cols[0])/(cols[1]-cols[0]))
    fluxes = [1, 10**(0.4 * 0.5)]
    assert color_key_func(fluxes) == cind

    for cind in range(33):
        col = cols[cind]
        for bind in range(nbands):
            true_fwhm = (0.8 + bind*0.2) * color_dep_fun(col)
            psf_img = galsim.ImageD(
                color_dep_mbobs[cind][bind][0].psf.image,
                scale=scale,
            )
            meas_fwhm = psf_img.calculateFWHM()
            assert np.allclose(true_fwhm, meas_fwhm, rtol=0, atol=0.05)
