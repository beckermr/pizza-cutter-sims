## tests of the pizza cutter

This directory stores runs of the sims to test the pizza cutter.

## results

| sim  | wcs  | gals | msk | psf | n_se | m [1e-3, 1-sigma]       | c [1e-5, 1-sigma]       |
| ---  | ---  | ---  | --- | --- | ---  | ---:                    | ---:                    |
| 0001 | d    | bg   | n/a | gf  | 1    |  0.425837 +/-  0.066487 |  0.072189 +/-  0.146772 |
| 0002 | ds   | bg   | n/a | gf  | 1    | -0.205942 +/-  0.121998 |  0.217237 +/-  0.142952 |
| 0003 | dse  | bg   | n/a | gf  | 1    |  0.451438 +/-  0.109515 | -0.265417 +/-  0.150538 |
| 0004 | de   | bg   | n/a | gf  | 1    |  0.396928 +/-  0.077780 |  0.209560 +/-  0.169468 |
| 0005 | dsep | bg   | n/a | gf  | 1    |  0.444695 +/-  0.062156 | -0.853732 +/-  0.301320 |


| sim  | wcs  | gals | msk | psf | n_se | m [1e-3, 3-sigma]       | c [1e-5, 3-sigma]       |
| ---  | ---  | ---  | --- | --- | ---  | ---:                    | ---:                    |
| 0017 | n/a  | bh   | n/a | gf  | 1    |  0.171208 +/-  0.578166 | -0.014212 +/-  1.198454 |


n_se:

 - number of SE images coadded

psf key:

 - g = Gaussian w/ FWHM of 0.9 arcsec
 - f = frac change in FWHM of +/-10%

wcs key:

 - d = dither SE images by +/- 1.5 pixels
 - s = change SE pixel scale by +/- 5%
 - e = shear SE coords by +/- 0.001
 - p = apply position angle rotation in [0, 360] degrees

gals key:

 - B = super bright mag 14 objects
 - b = bright mag 18 exp objects
 - f = faint mag 23.5 objects
 - g = gals in grid
 - r = gals positioned randomly
 - d = DES-like objects
 - l = LSST-like objects
 - h = gals in hexagonal grid w/ rotations
 - s = glass-like galaxy layout

msk key:

 - r = cosmic rays
 - c = bad columns
 - s = streaks
 - t = interpolate bad regions
 - o{N} = set ormask flag from region of size 2N about object center
 - k{N} = only skinny bad columns (e.g. no bleed masks) of max width N
 - a{N} = use all rotation angles for masks (a8) or just all 90 degree rotations (a4)
 - e{N} = expand star masks by N pixels

shear key:

 - w = swap role of shears 1 and 2

coadd key:

 - {N} = coadd this many images
 - None = skip coadding

stars key:

 - g{N} = GAIA-like at approx dens of N per arcmin^2
 - n = skip the interpolation inside regions with star holes
 - e = put star flags in metadetect bmask_flags for skipping stuff on survey edges
 - f = when interpolating fill with noise
 - a = apodize as opposed to interpolating

psf key:
 - m1{p|m}{N.NN} = shear has mean of N.NN in 1-axis with sign p or m
 - m2{p|m}{N.NN} = shear has mean of B.NN in 2-axis with sign p or m

bkg key:
 - c = constant level as listed in config

## notes

### run0027
  - when done with mfrac and ormak/bmask at the sheared locations had 1.5 +/ 0.5 10^-3 bias

### LSST pixel scale runs

  run0042_wcs-dse_gals-lr_msk-rcste16_coadd-1_stars-ag1
  run0043_wcs-dse_gals-lr_msk-None_coadd-1
  run0045_wcs-None_gals-lr_msk-None_coadd-None
  run0046_wcs-None_gals-Bh_msk-None_coadd-None
  run0047_wcs-None_gals-Br_msk-None_coadd-None
  run0048_wcs-None_gals-fr_msk-None_coadd-None
  run0049_wcs-None_gals-Br_msk-None_coadd-None_psf-k
  run0050_wcs-None_gals-fr_msk-None_coadd-None_psf-k
