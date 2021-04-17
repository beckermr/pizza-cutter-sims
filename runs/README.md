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

 - g = Gaussian w/ FWHM of 0.8 arcsec
 - f = frac change in FWHM of +/-10%

wcs key:

 - d = dither SE images by +/- 1.5 pixels
 - s = change SE pixel scale by +/- 5%
 - e = shear SE coords by +/- 0.001
 - p = apply position angle rotation in [0, 360] degrees

gals key:

 - b = bright mag 18 exp objects
 - f = faint mag 23.5 objects
 - g = gals in grid
 - r = gals positioned randomly
 - d = DES-like objects
 - h = gals im hexagonal grid w/ rotations
 - s = glass-like galaxy layout

msk key:

 - r = cosmic rays
 - c = bad columns
 - t = interpolate bad regions
 - o{N} = set ormask flag from region of size 2N about object center
 - k{N} = only skinny bad columns (e.g. no bleed masks) of max width N
 - a{N} = use all rotation angles for masks (a8) or just all 90 degree rotations (a4)

shear key:

 - w = swap role of shears 1 and 2
