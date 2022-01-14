import os
import time
import uuid
import subprocess
import cloudpickle
import joblib
import atexit
import numpy as np

from concurrent.futures import ThreadPoolExecutor

ALL_CONDOR_JOBS = {}

STATUS_DICT = {
    "": "unknown condor failure :(",
    "1": "Idle",
    "2": "Running",
    "3": "Removed",
    "4": "Completed",
    "5": "Held",
    "6": "Transferring Output",
    "7": "Suspended",
}


def _kill_condor_jobs():
    chunksize = 100
    cjobs = []
    for cjob in list(ALL_CONDOR_JOBS):
        cjobs.append(cjob)
        if len(cjobs) == chunksize:
            _cjobs = " ".join(cjobs)
            os.system("condor_rm " + _cjobs)
            os.system("condor_rm -forcex " + _cjobs)
            cjobs = []

    if cjobs:
        os.system("condor_rm " + _cjobs)
        os.system("condor_rm -forcex " + _cjobs)
        cjobs = []


atexit.register(_kill_condor_jobs)


def _submit_and_poll_function(
    execid, execdir, id, poll_interval, max_poll_time, func, args, kwargs
):
    infile = os.path.abspath(os.path.join(execdir, id, "input.pkl"))
    outfile = os.path.abspath(os.path.join(execdir, id, "output.pkl"))
    logfile = os.path.abspath(os.path.join(execdir, id, "log.oe"))
    condorfile = os.path.join(execdir, id, "condor.sub")

    os.makedirs(os.path.join(execdir, id), exist_ok=True)

    ##############################
    # dump the file
    with open(infile, "wb") as fp:
        cloudpickle.dump(joblib.delayed(func)(*args, **kwargs), fp)

    ##############################
    # submit the condor job
    with open(condorfile, "w") as fp:
        fp.write(
            """\
Universe = vanilla
Notification = Never
# Run this exe with these args
Executable = %s
# Image_Size =  2500000
request_memory = 2G
kill_sig = SIGINT
+Experiment = "astro"

+job_name = "%s"
Arguments = %s %s %s
Queue
""" % (
                os.path.join(execdir, "run.sh"),
                "job-%s-%s" % (execid, id),
                infile,
                outfile,
                logfile,
            ),
        )

    sub = subprocess.run(
        "condor_submit %s" % condorfile,
        shell=True,
        check=True,
        capture_output=True,
    )

    cjob = None
    for line in sub.stdout.decode("utf-8").splitlines():
        line = line.strip()
        if "submitted to cluster" in line:
            line = line.split(" ")
            cjob = line[5] + "0"
            break

    assert cjob is not None
    ALL_CONDOR_JOBS[cjob] = None

    ##############################
    # poll for it being done
    check_file = outfile + ".done"
    status_code = None
    timed_out = False
    start_poll = time.time()
    while not os.path.exists(check_file):
        time.sleep(poll_interval)

        res = subprocess.run(
            "condor_q %s -af JobStatus" % cjob,
            shell=True,
            capture_output=True,
        )
        status_code = res.stdout.decode("utf-8").strip()
        if status_code in ["3", "5", "7"]:
            break

        if time.time() - start_poll < max_poll_time:
            timed_out = True
            break

    del ALL_CONDOR_JOBS[cjob]
    if os.path.exists(outfile):
        res = joblib.load(outfile)
    elif status_code in ["3", "5", "7"]:
        res = RuntimeError(
            "Condor job %s: status %s" % (id, STATUS_DICT[status_code])
        )
    elif timed_out:
        res = RuntimeError("Condor job %s: timed out after %ss!" % (id, max_poll_time))
    else:
        res = RuntimeError("Condor job %s: no status or job output found!" % id)

    subprocess.run(
        "rm -f %s %s %s.done" % (infile, outfile, outfile),
        shell=True,
        check=True,
    )

    if isinstance(res, Exception):
        raise res
    else:
        return res


class CondorExecutor():
    _worker_init = """\
#!/bin/bash

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

run-pickled-task $1 $2 $3 >& ${tmpdir}/$(basename $3)

mv ${tmpdir}/$(basename $3) $3
"""

    def __init__(
        self, max_workers=10000, poll_interval=10, conda_env="pizza-cutter-sims",
        verbose=None, job_timeout=7200,
    ):
        self.max_workers = max_workers
        self.execid = uuid.uuid4().hex
        self.execdir = "condor-exec/%s" % self.execid
        self.conda_env = conda_env
        self._exec = None
        self.poll_interval = poll_interval
        self.job_timeout = job_timeout

    def __enter__(self):
        os.makedirs(self.execdir, exist_ok=True)
        print("starting condor executor: %s" % self.execdir, flush=True)

        with open(os.path.join(self.execdir, "run.sh"), "w") as fp:
            fp.write(self._worker_init % self.conda_env)
        subprocess.run(
            "chmod u+x " + os.path.join(self.execdir, "run.sh"),
            shell=True,
            check=True,
        )
        self._exec = ThreadPoolExecutor(max_workers=self.max_workers)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._exec.shutdown()
        self._exec = None

    def submit(self, func, *args, **kwargs):
        subid = uuid.uuid4().hex
        fut = self._exec.submit(
            _submit_and_poll_function,
            self.execid,
            self.execdir,
            subid,
            min(self.poll_interval * max(1, np.sqrt(len(ALL_CONDOR_JOBS)/100)), 300),
            self.job_timeout,
            func,
            args,
            kwargs,
        )
        fut.execid = self.execid
        fut.subid = subid
        return fut
