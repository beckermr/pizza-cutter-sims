## tests of the pizza cutter

This directory stores runs of the sims to test the pizza cutter.

## results

| sim  | wcs  | gals | msk | m [1e-3, 1-sigma]       | c [1e-5, 1-sigma]       |
| ---  | ---  | ---  | --- | ---:                    | ---:                    |
| 0001 | d    | bg   | n/a |  0.425837 +/-  0.066487 |  0.072189 +/-  0.146772 |
| 0002 | ds   | bg   | n/a | -0.205942 +/-  0.121998 |  0.217237 +/-  0.142952 |
| 0003 | dse  | bg   | n/a |  0.451438 +/-  0.109515 | -0.265417 +/-  0.150538 |
| 0004 | de   | bg   | n/a |  0.396928 +/-  0.077780 |  0.209560 +/-  0.169468 |
| 0005 | dsep | bg   | n/a |  0.444695 +/-  0.062156 | -0.853732 +/-  0.301320 |

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

msk key:

 - r = cosmic rays
 - c = bad columns