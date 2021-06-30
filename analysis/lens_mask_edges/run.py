#!/usr/bin.env python
from mdet_tools import meas_m, format_mc_res
import sys
import click


@click.command()
@click.option(
    '--n-stars', default=20, type=int, required=True,
    help='# of stars per image',
)
@click.option(
    '--sym', default="3", type=str, required=True,
    help=(
        'how to symmtrize the image. either a comma-separate list of angles in '
        'degrees, an int specifying the number of 90 degree rotations, '
        'or 0 for no symmetrization.'
    )
)
@click.option(
    '--seed', default=None, type=int, required=True,
    help="the RNG seed",
)
@click.option(
    '--mask-width', default=None, type=int, required=True,
    help="the half-width of the region about an object's center that cannot be masked",
)
@click.option(
    '--n-sims', default=5_000_000, type=int,
    help="the number of sims to run",
)
def main(n_stars, sym, seed, mask_width, n_sims):
    if "," in sym:
        sym = [int(s) for s in sym.split(",")]
    else:
        sym = int(sym)
    print("sim properties:")
    print("    star density: %s [num per arcmin^2]" % (n_stars / (235*0.20/60)**2))
    print("    sym:", sym)
    print("    seed:", seed)
    print("    mask_width:", mask_width)
    print(" ")
    sys.stdout.flush()

    res = meas_m(
        n_stars=n_stars, mask_width=mask_width, seed=seed, n_jobs=n_sims,
        sym=sym,
    )

    print("\nsim results:")
    print("    star density: %s [num per arcmin^2]" % (n_stars / (235*0.20/60)**2))
    print("    sym:", sym)
    print("    seed:", seed)
    print("    mask_width:", mask_width)
    print("    " + format_mc_res(res, space=4))
    sys.stdout.flush()


if __name__ == "__main__":
    main()
