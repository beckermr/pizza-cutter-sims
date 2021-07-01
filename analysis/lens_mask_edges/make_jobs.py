JOB_TMP = """\
#!/bin/bash

#SBATCH --job-name={job_name}
#SBATCH --account=metashear
#SBATCH --partition=bdwall
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --output=log_{job_name}-%j.oe
#SBATCH --time=48:00:00

source ~/.bashrc
conda activate pizza-cutter-sims

python run.py --seed={seed} --n-sims={n_sims} --mask-width={mask_width} --sym={sym}
"""

# seeds = [10, 213423, 67465, 12432, 343]
# mws = [8, 4, 2, 2, 2]
# syms = [0, 0, 0, 1, 3]
# n_sims = 5_000_000

seeds = [665]
mws = [2]
syms = [8]
n_sims = 5_000_000


for seed, mw, sym in zip(seeds, mws, syms):
    job_name = "mw%d_sym%d_seed%d_nsims%d" % (
        mw,
        sym,
        seed,
        n_sims,
    )
    cmd = JOB_TMP.format(
        seed=seed,
        n_sims=n_sims,
        mask_width=mw,
        sym=sym,
        job_name=job_name,
    )
    with open("job_%s.sh" % job_name, "w") as fp:
        fp.write(cmd)
