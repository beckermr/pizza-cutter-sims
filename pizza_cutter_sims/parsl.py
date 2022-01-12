import logging
import atexit

import parsl
from parsl.providers import CondorProvider

from parsl.config import Config
from parsl.executors import HighThroughputExecutor

from esutil.pbar import PBar

logger = logging.getLogger(__name__)


def _kill_all_jobs(condor_provider):
    job_ids = list(condor_provider.resources.keys())
    condor_provider.cancel(job_ids)


class MyCondorProvider(CondorProvider):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        atexit.register(_kill_all_jobs, self)


class ParslCondorPool():
    """A schwimbad pool-type interface to parsl.

    Parameters
    ----------
    conda_env : str, optional
        The conda environmant to activate. Default is "pizza-cutter-sims".
    walltime_hours : int, optional
        The maximum walltime of a condor job. Default is 12.
    parallelism : float, optional
        How quickly to scale up condor jobs. A value of 1 will lanuch one condor
        job for every mapped process. Default is 0.75.
    verbose : int
        If greater than 0, print a progress bar.
    """

    _condor_preamble = """\
Universe = vanilla
Notification = Never
request_memory = 2G
kill_sig = SIGINT
+Experiment = "astro"
"""

    _worker_init = """\
source ~/.bashrc

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

if [[ -n $_CONDOR_SCRATCH_DIR ]]; then
    # the condor system creates a scratch directory for us,
    # and cleans up afterward
    tmpdir=$_CONDOR_SCRATCH_DIR
    export TMPDIR=$tmpdir
else
    # otherwise use the TMPDIR
    tmpdir='.'
    mkdir -p $tmpdir
fi

source activate %s
"""

    def __init__(
        self,
        conda_env="pizza-cutter-sims",
        walltime_hours=12,
        parallelism=0.75,
        verbose=0,
    ):
        self._conda_env = conda_env
        self._walltime_hours = walltime_hours
        self._parallelism = parallelism
        self._verbose = verbose
        self._active = False
        self.config = Config(
            strategy='htex_auto_scale',
            executors=[
                HighThroughputExecutor(
                    worker_debug=True,
                    max_workers=1,
                    poll_period=10000,
                    provider=MyCondorProvider(
                        cores_per_slot=1,
                        # mem_per_slot=2, done in scheduler_options
                        nodes_per_block=1,
                        init_blocks=0,
                        parallelism=self._parallelism,
                        max_blocks=10000,  # 10 seconds in milliseconds
                        scheduler_options=self._condor_preamble,
                        worker_init=self._worker_init % self._conda_env,
                        walltime="%d:00:00" % self._walltime_hours,
                        cmd_timeout=300,  # five minutes
                    )
                )
            ],
        )

    def __enter__(self):
        parsl.load(self.config)
        self._active = True
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        parsl.clear()
        self._active = False

    def map(self, func, args):
        """Map a function over a list of arguments."""
        if not self._active:
            raise RuntimeError("You must call 'map' within a context manager!")

        pfunc = parsl.python_app(func)

        futs = [pfunc(arg) for arg in args]

        results = []
        n_tot = len(futs)
        itr = range(n_tot)
        if self._verbose > 0:
            itr = PBar(itr, n_bars=79, desc='parsl+condor map')
        for i in itr:
            fut = futs[i]
            # this call sometimes helps the result return properly
            fut.done()
            results.append(fut.result())

        return results
